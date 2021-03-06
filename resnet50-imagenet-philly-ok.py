import tarfile
from torch.utils import data
import io
from PIL import Image

import os
import sys
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.datasets
import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
from torch.multiprocessing import Process
from random import Random
from math import ceil
from torch.nn.parallel import DistributedDataParallel as DDP
from time import perf_counter as pc
import argparse

class ImageTarDataset(data.Dataset):

    def __init__(self, tar_file, transform=None):
        # time0 = pc()
        self.tar_file = tar_file
        self.tar_handle = None
        categories_set = set()
        self.tar_members = []
        self.categories = {}
        with tarfile.open(tar_file, 'r:') as tar:
            # time1 = pc()
            for tar_member in tar.getmembers():
                if tar_member.name.count('/') != 2:
                    continue

                categories_set.add(self.get_category_from_filename(tar_member.name))
                self.tar_members.append(tar_member)
            # time2 = pc()
        index = 0
        categories_set = sorted(categories_set)
        for category in categories_set:
            self.categories[category] = index
            index += 1
        self.transform = transform
        # print("tarfile.open: ", time1 - time0, " categorization:", time2-time1, " sorting: ", pc() - time2)

    def get_category_from_filename(self, filename):
        begin = filename.find('/')
        begin += 1
        end = filename.find('/', begin)
        return filename[begin:end]

    def __len__(self):
        return len(self.tar_members)

    def __getitem__(self, index):
        if self.tar_handle is None:
            self.tar_handle = tarfile.open(self.tar_file, 'r:')

        sample = self.tar_handle.extractfile(self.tar_members[index])
        sample = Image.open(sample)
        sample = sample.convert('RGB')
        if self.transform:
            sample = self.transform(sample)
        category = self.categories[self.get_category_from_filename(self.tar_members[index].name)]
        return sample, category


class ImageFolderDataset(data.Dataset):

    def __init__(self, root, transform=None):

        self.transform = transform
        categories_set = [d.name for d in os.scandir(root) if d.is_dir()]
        categories_set.sort()
        index = 0
        self.categories = {}
        self.files = []
        for category in categories_set:
            self.categories[category] = index
            index += 1

        for target in self.categories.keys():
            d = os.path.join(root, target)
            if not os.path.isdir(d):
                continue
            for r, _, file_names in os.walk(d):
                for file_name in sorted(file_names):
                    file_path = os.path.join(r, file_name)
                    item = (file_path, self.categories[target])
                    self.files.append(item)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path, target = self.files[index]
        with open(path, 'rb') as f:
            sample = Image.open(f)
            sample = sample.convert('RGB')
            if self.transform:
                sample = self.transform(sample)
            return sample, target


def make_imagenet_dataset(data_loader_name, train=True):
    tail = ""
    if data_loader_name == "ImageFolderDataset":
        data_loader = ImageFolderDataset
    elif data_loader_name == "ImageTarDataset":
        data_loader = ImageTarDataset
        tail = ".tar"

    # print("Dataloader: ", data_loader_name)
    def imagenet_train_dataset(data_path, batch_size, num_workers):
        train_data_path = data_path + "/train" + tail
        train_dataset = data_loader(
            train_data_path,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]))

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

        train_set = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size,
            pin_memory=False, num_workers=num_workers,
            shuffle=False, sampler=train_sampler
        )

        return train_set, train_sampler, len(train_dataset)

    def imagenet_val_dataset(data_path, batch_size, num_workers):
        val_data_path = data_path + "/val" + tail

        val_dataset = data_loader(
            val_data_path,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]))
        val_set = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size,
            pin_memory=False, num_workers=num_workers,
            shuffle=False)

        return val_set

    if train == True:
        return imagenet_train_dataset
    else:
        return imagenet_val_dataset


def adjust_learning_rate(optimizer, epoch, init_learning_rate):
    lr = init_learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def run(data_loader_name, batch_size):
    num_workers = 4
    data_path = os.environ['DATA_DIR']
    output_path = os.environ['OUTPUT_DIR']
    output_path += os.environ['PHILLY_JOB_ID']
    output_path += os.environ['PHILLY_CONTAINER_INDEX']
    init_learning_rate = 0.128
    torch.manual_seed(1234)
    with open(output_path, 'w') as f:
        sys.stdout = f
        print("myrank: ", dist.get_rank(), " device_count: ", torch.cuda.device_count())
        imagenet_train_dataset = make_imagenet_dataset(data_loader_name, train=True)
        train_set, train_sampler, data_size = imagenet_train_dataset(data_path, batch_size, num_workers)
        gpu_id = dist.get_rank() % torch.cuda.device_count()
        torch.cuda.set_device(gpu_id)
        model = models.__dict__["resnet50"]()
        model.cuda(torch.cuda.current_device())
        model = DDP(model, [gpu_id])
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = optim.SGD(
            model.parameters(), lr=init_learning_rate,
            momentum=0.875, weight_decay=3.0517578125e-05
        )

        for epoch in range(10):
            model.train()
            train_sampler.set_epoch(epoch)
            adjust_learning_rate(optimizer, epoch, init_learning_rate)
            epoch_loss = 0.0

            accumulated = float()
            accumulated2 = float()
            time0 = pc()
            for idx, (data, target) in enumerate(train_set):
                # pass
                time1 = pc()
                data = data.cuda()
                target = target.cuda()
                time2 = pc()
                output = model(data)
                loss = criterion(output, target)
                epoch_loss += loss.data.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                accumulated2 += pc() - time2
                accumulated += pc() - time1

            diff = pc() - time0
            tensor_diff = torch.tensor([diff])
            tensor_diff = tensor_diff.cuda()
            dist.all_reduce(tensor_diff.data, op=dist.ReduceOp.SUM)

            tensor_accumulated = torch.tensor([accumulated])
            tensor_accumulated = tensor_accumulated.cuda()
            dist.all_reduce(tensor_accumulated.data, op=dist.ReduceOp.SUM)

            tensor_accumulated2 = torch.tensor([accumulated2])
            tensor_accumulated2 = tensor_accumulated2.cuda()
            dist.all_reduce(tensor_accumulated2.data, op=dist.ReduceOp.SUM)

            if dist.get_rank() == 0:
                diff = (tensor_diff.data / float(dist.get_world_size()))
                accumulated = (tensor_accumulated.data / float(dist.get_world_size()))
                accumulated2 = (tensor_accumulated2.data / float(dist.get_world_size()))
                diff = diff.tolist()[0]
                accumulated = accumulated.tolist()[0]
                accumulated2 = accumulated2.tolist()[0]
                rate_total = int(data_size / diff)
                rate_gpu_bus = int(data_size / accumulated)
                rate_gpu = int(data_size / accumulated2)
                elapsed_time = round(accumulated / len(train_set) * 1000, 2)
                print("Epoch: ", epoch, ", total rate: ", rate_total, " images/sec",
                      " GPU and bus rate: ", rate_gpu_bus, " images/sec",
                      " GPU only rate:", rate_gpu, " images/sec", " elapsed time per batch: (ms)", elapsed_time)




def init_processes(master, port, data_loader_name, batch_size, global_rank, size, fn, backend='nccl'):
    os.environ['MASTER_ADDR'] = master
    os.environ['MASTER_PORT'] = port
    # print("world: ", size, " rank: ", global_rank)
    dist.init_process_group(backend, rank=global_rank, world_size=size)
    fn(data_loader_name, batch_size)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--world", type=int)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--master", type=str)
    parser.add_argument("--port", default=14007, type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--batch", type=int)
    args = parser.parse_args()

    world_size = args.world
    node_id = args.rank
    node_size = args.gpu
    batch_size = args.batch
    processes = []
    for local_rank in range(node_size):
        p = Process(target=init_processes, args=(
            args.master, args.port, args.dataset, batch_size, (node_id * node_size) + local_rank, node_size * world_size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()