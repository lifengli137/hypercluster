### no pre-decoding
print("No pre-decoding")
DUPLICATION = 16

import os
from time import perf_counter as pc
from PIL import Image
import sys
import io
from array import array
import psutil
import pickle
from math import ceil
from os import cpu_count
from multiprocessing import Pool, current_process
    

# Global variables
# TODO: Dependency injection
dataset = "train"
root = "./" + dataset
mem_used=[]
mem_used.append(psutil.virtual_memory().used/1024**3)

def create_dat(category_subset):
    total_file_count = 0
    for index, category in category_subset:
        d = os.path.join(root, category)
        for r, _, file_names in os.walk(d):
            total_file_count += len(file_names)


    process_start = pc()
    process_id = current_process().pid
    metadatas = []
    with open(f'./wills-nodecoding/{dataset}.data.{process_id}', 'wb') as imageOut:
        count = 0
        offset = 0
        
        run_start_time = pc()
        last_batch_start_time = pc()
        
        for index,target in category_subset:
            d = os.path.join(root, target)
            if not os.path.isdir(d):
                continue
            for r, _, file_names in os.walk(d): # Root, directory, files
                for file_name in file_names:
                    file_path = os.path.join(r, file_name)
                    #print(file_path)
                    with open(file_path, "rb") as fd:
                        image_bytes = fd.read()
                        imageOut.write(image_bytes)
                        metadatas.append(process_id)
                        metadatas.append(offset)
                        metadatas.append(len(image_bytes))
                        metadatas.append(index)

                    offset += len(image_bytes)
                    batch_size = 1000
                    if count % batch_size == 0 and count > 0:
                        mem = psutil.virtual_memory()
                        b_elapsed = pc() - last_batch_start_time  #Seconds elapsed since batch start
                        b_imps = batch_size / b_elapsed  # Images per second for this batch

                        t_elapsed = pc() - run_start_time  # Elapsed time since start of all batches
                        t_imps = count / t_elapsed  # Average images per second of all batches
                        
                        cper = count / total_file_count  # Percent completion through all files. AKA progress

                        mem_string = f'MU:{mem.percent:5}% - F:{mem.free/1024**3:5.1f}G - A:{mem.available/1024**3:5.1f}G - U:{mem.used/1024**3:5.1f}G'

                        print(f'{count:10,} ({cper*100:2.0f}%) - {mem_string} - {b_imps:3,.0f} IM/s this batch - {t_imps:3,.0f} tot IM/s')
                        mem_used.append(mem.used/1024**3)
                        last_batch_start_time = pc()
                    count += 1
    process_end = pc()
    elapsed = process_end - process_start
    rate = count / elapsed
    print(f"Process {process_id} completed its {count:,} images in {elapsed:,.2f}s ({rate:,.2f}im/s)")
    return metadatas
    
def load():    
    
    categories_set = [d.name for d in os.scandir(root) if d.is_dir()]
    categories_set.sort()
    tmp = []
    for _ in range(DUPLICATION):
        tmp.extend(categories_set)
    categories_set = tmp
    
    index = 0

    list_cats = list(enumerate(categories_set))

    # Batch up categories for multiprocessing
    physical_cores = cpu_count()  # Hopefully accounts for hyperthreading
    batch_size = int(ceil(len(list_cats) / (physical_cores)))
    #print(f"Found {len(list_cats)} categories in dataset {dataset}.")
    print(f"Detected {cpu_count()} logical cores, which should be {physical_cores} physical cores, resulting in a batch size of {batch_size} categories per core.")
    batches = []

    for i in range(physical_cores):
        if (i+1) * batch_size > len(list_cats):
            batches.append(list_cats[(i) * batch_size:]) # Not elegant. TODO: make better
        else:
            batches.append(list_cats[(i) * batch_size: (i + 1) * batch_size])

    # Run each batch in parallel
    process_pool = Pool(physical_cores)
    results = process_pool.map(create_dat, batches) # Metadata for images
    # Collapse multi-layered list levels
    joined_metadata = []
    for md_set in results:
        joined_metadata.extend(md_set)

    metadatas = array("Q", joined_metadata)
    
    print(len(metadatas), metadatas[0], metadatas[1], metadatas[2], metadatas[3], metadatas[4], metadatas[5], metadatas[6])
    with open(f'./wills-nodecoding/{dataset}.metadata', 'wb') as metadataOut:
        pickle.dump(metadatas,metadataOut)
    
if __name__ == "__main__":  # Prevents... recursion, probably.
    start = pc()
    load()
    finish = pc()
    print(f"Finished loading, took {finish - start:,.1f} seconds.")
