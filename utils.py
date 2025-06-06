import os

import nvidia_smi
import pickle

def query_memory(verbose=False):
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    # card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate

    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

    bytes_to_GB = 10**9
    total, free, used = info.total / bytes_to_GB, info.free / bytes_to_GB, info.used / bytes_to_GB
    nvidia_smi.nvmlShutdown()
    
    if verbose:
        print("Total memory {:.2f} GB, free {:.2f} GB, used {:.2f} GB".format(total, free, used))
    return total, free, used


def save_checkpoint(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_checkpoint(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return None

def purge_checkpoint(filename):
    if os.path.exists(filename):
        os.remove(filename)
    return None