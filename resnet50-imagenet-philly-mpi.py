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
import torch.cuda.nvtx as nvtx
import numpy as np
from array import array
import pickle

pin_memory = False
shared_memory = False

class _RepeatSampler(object):

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class BetterDataLoader(torch.utils.data.dataloader.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()
 

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            #yield next(self.iterator)
            next_input, next_target = next(self.iterator)
            if pin_memory:
                next_input = next_input.pin_memory()
                next_target = next_target.pin_memory()
            yield next_input, next_target


class SoftwarePipeline(object):


    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.stream = None

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        if self.stream is None:
            self.stream = torch.cuda.Stream()
        first = True

        for next_input, next_target in self.dataloader:
            with torch.cuda.stream(self.stream):
                next_input = next_input.cuda(non_blocking=True)
                next_target = next_target.cuda(non_blocking=True)
            if not first:
                yield input, target
            else:
                first = False

            torch.cuda.current_stream().wait_stream(self.stream)
            input = next_input
            target = next_target

        yield input, target


class ImageModes(object):
    def __init__(self):
        self.list_of_modes = ['1', 'L', 'P', 'RGB', 'RGBA', 'CMYK', 'YCbCr', 'LAB', 'HSV', 'I', 'F']
        self.dict_of_modes = {}
        for index, mode in enumerate(self.list_of_modes):
            self.dict_of_modes[mode] = index
            
    def get_mode_by_index(self, index):
        return self.list_of_modes[index]
    
    def get_index_by_mode(self, mode):
        return self.dict_of_modes[mode]

class InMemoryImageDataset(data.Dataset):

    def __init__(self, root, transform=None):



        self.modes = ImageModes()
        
        self.transform = transform
        metadata_path = root + ".metadata"
        with open(metadata_path, 'rb') as metadataIn:
            self.metadatas = pickle.load(metadataIn)
        self.data_path = root + ".data"
        with open(self.data_path, 'rb') as imageIn:
            self.images = bytes(imageIn.read())


    def __len__(self):
        return int(len(self.metadatas)/6)

    def __getitem__(self, index):
        OFFSET = 0
        SIZE = 1
        LABEL = 2
        X = 3
        Y = 4
        MODE = 5
        STRIP = 6
        
        offset = STRIP * index + OFFSET
        size = STRIP * index + SIZE
        label = STRIP * index + LABEL
        x = STRIP * index + X
        y = STRIP * index + Y
        mode = STRIP * index + MODE
        

        image = Image.frombytes(mode=self.modes.get_mode_by_index(self.metadatas[mode]), size=(self.metadatas[x], self.metadatas[y]), data=self.images[self.metadatas[offset]:self.metadatas[offset]+self.metadatas[size]])
        image = image.convert('RGB')
        if self.transform:
            image_tensor = self.transform(image)
            
        
        label_tensor = torch.tensor(self.metadatas[label])
        if shared_memory:
            image_tensor.share_memory_()
            label_tensor.share_memory_()
        
        return image_tensor, label_tensor

class IndexInMemoryImageInDiskDataset(data.Dataset):

    def __init__(self, root, transform=None):


        self.modes = ImageModes()
        
        self.transform = transform
        metadata_path = root + ".metadata"
        with open(metadata_path, 'rb') as metadataIn:
            self.metadatas = pickle.load(metadataIn)
        self.data_path = root + ".data"
        self.imageIn = None

    def __len__(self):
        return int(len(self.metadatas)/6)

    def __getitem__(self, index):
        if self.imageIn is None:
            self.imageIn = open(self.data_path, 'rb')
            
        OFFSET = 0
        SIZE = 1
        LABEL = 2
        X = 3
        Y = 4
        MODE = 5
        STRIP = 6
        
        offset = STRIP * index + OFFSET
        size = STRIP * index + SIZE
        label = STRIP * index + LABEL
        x = STRIP * index + X
        y = STRIP * index + Y
        mode = STRIP * index + MODE
        
        self.imageIn.seek(self.metadatas[offset])
        
        image = self.imageIn.read(self.metadatas[size])
        
        
        image = Image.frombytes(mode=self.modes.get_mode_by_index(self.metadatas[mode]), size=(self.metadatas[x], self.metadatas[y]), data=image)
        image = image.convert('RGB')
        if self.transform:
            image_tensor = self.transform(image)
            image_tensor.share_memory_()
        
        label_tensor = torch.tensor(self.metadatas[label])
        label_tensor.share_memory_()
        return image_tensor, label_tensor

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
            image_tensor = self.transform(sample)
            image_tensor.share_memory_()
        category = self.categories[self.get_category_from_filename(self.tar_members[index].name)]
        label_tensor = torch.tensor(category)

        if shared_memory:
            image_tensor.share_memory_()
            label_tensor.share_memory_()

        return image_tensor, label_tensor


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
                image_tensor = self.transform(sample)
                image_tensor.share_memory_()
        label_tensor = torch.tensor(target)

        if shared_memory:
            image_tensor.share_memory_()
            label_tensor.share_memory_()
        return image_tensor, label_tensor


def make_imagenet_dataset(dataset_name, train=True, dataloader=False):

    if dataloader:
        dataloader = BetterDataLoader
    else:
        dataloader = torch.utils.data.DataLoader


    tail = ""
    if dataset_name == "ImageFolderDataset":
        dataset = ImageFolderDataset
    elif dataset_name == "ImageTarDataset":
        dataset = ImageTarDataset
        tail = ".tar"
    elif dataset_name == "InMemoryImageDataset":
        dataset = InMemoryImageDataset
    elif dataset_name == "IndexInMemoryImageInDiskDataset":
        dataset = IndexInMemoryImageInDiskDataset

    
    # print("Dataloader: ", dataset)
    def imagenet_train_dataset(data_path, batch_size, num_workers):
        train_data_path = data_path + "/train" + tail
        train_dataset = dataset(
            train_data_path,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]))

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

        train_set = dataloader(
            train_dataset, batch_size=batch_size,
            pin_memory=False, num_workers=num_workers,
            shuffle=False, sampler=train_sampler
        )

        return train_set, train_sampler, len(train_dataset)

    def imagenet_val_dataset(data_path, batch_size, num_workers):
        val_data_path = data_path + "/val" + tail

        val_dataset = dataset(
            val_data_path,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]))
        val_set = dataloader(
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


def get_global_rank():
    if os.environ.get('OMPI_COMM_WORLD_RANK') is not None: 
        return int(os.environ.get('OMPI_COMM_WORLD_RANK') or 0)
    else:
        return 0

def get_global_size():
    if os.environ.get('OMPI_COMM_WORLD_SIZE') is not None:
        return int(os.environ.get('OMPI_COMM_WORLD_SIZE'))
    else:
        return 1



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ImageTarDataset")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pin_memory", type=int, default=0)
    parser.add_argument("--shared_memory", type=int, default=0)
    parser.add_argument("--pipeline", type=int, default=0)
    parser.add_argument("--dataloader", type=int, default=0)
    args = parser.parse_args()
    pin_memory = args.pin_memory
    shared_memory = args.shared_memory

    world_size = get_global_size()
    world_rank = get_global_rank()
    dataset = args.dataset
    batch_size = args.batch
    print(os.environ['MASTER_ADDR'], " ", os.environ.get('MASTER_PORT'))
    print("world_size: ", world_size, " world_rank: ", world_rank," dataset:", dataset) 
    dist.init_process_group(backend='nccl', rank=world_rank, world_size=world_size)
    num_workers = args.workers
    init_learning_rate = 0.128
    torch.manual_seed(1234)
    imagenet_train_dataset = make_imagenet_dataset(dataset, train=True, dataloader=args.dataloader)
    data_path = os.environ['DATA_DIR']
    train_set, train_sampler, data_size = imagenet_train_dataset(data_path, batch_size, num_workers)

    if args.pipeline:
        train_set = SoftwarePipeline(train_set)

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
        nvtx.range_push('epoch')
        
        nvtx.range_push('set_train')
        model.train()
        nvtx.range_pop() # set train
        
        nvtx.range_push('set_epoch')
        train_sampler.set_epoch(epoch)
        nvtx.range_pop() # set epoch
        
        nvtx.range_push('adjust_lr')
        adjust_learning_rate(optimizer, epoch, init_learning_rate)
        nvtx.range_pop() # adjust lr

        time0 = pc()
        
        for idx, (data, target) in enumerate(train_set):
            nvtx.range_push('iteration')

            nvtx.range_push('copy')
            data = data.cuda()
            target = target.cuda()
            nvtx.range_pop() # copy

            nvtx.range_push('forward')
            output = model(data)
            nvtx.range_pop() # forward

            nvtx.range_push('loss')
            loss = criterion(output, target)
            nvtx.range_pop() # loss

            nvtx.range_push('zero')
            optimizer.zero_grad()
            nvtx.range_pop() # zero

            nvtx.range_push('backward')
            loss.backward() 
            nvtx.range_pop() # backward

            nvtx.range_push('optimizer')
            optimizer.step()
            nvtx.range_pop() # optimizer

            nvtx.range_pop() # per iteration

        if dist.get_rank() == 0:
            print(f'Epoch: {epoch:8} - Loss: {loss:5.2f} - Througput: {1/((pc() - time0)/data_size): 8.2f} images/sec')

        nvtx.range_pop() # per epoch
