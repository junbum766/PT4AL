import faiss
import numpy as np
import torch.backends.cudnn as cudnn
import os
import torchvision
import torchvision.transforms as transforms
from models import *
from loader import TypiLoader
from utils import progress_bar


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "2"  # Set the GPUs 3 to use

device = 'cuda' if torch.cuda.is_available() else 'cpu'


transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = TypiLoader()
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)


def get_nn(features, num_neighbors):
    # calculates nearest neighbors on GPU
    d = features.shape[1]
    features = features.astype(np.float32)
    cpu_index = faiss.IndexFlatL2(d)
    gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
    gpu_index.add(features)  # add vectors to the index
    distances, indices = gpu_index.search(features, num_neighbors + 1)
    # 0 index is the same sample, dropping it
    return distances[:, 1:], indices[:, 1:]


def get_mean_nn_dist(features, num_neighbors, return_indices=False):
    distances, indices = get_nn(features, num_neighbors)
    mean_distance = distances.mean(axis=1)
    if return_indices:
        return mean_distance, indices
    return mean_distance


def calculate_typicality(features, num_neighbors):
    mean_distance = get_mean_nn_dist(features, num_neighbors)
    # low distance to NN is high density
    typicality = 1 / (mean_distance + 1e-5)
    return typicality


if __name__ == "__main__":
    class_num = 10
    simclr_feature = np.load("/home/ubuntu/junbeom/repo/PT4AL/simCLR_feature/features_seed1.npy", allow_pickle=True)
    typicality = calculate_typicality(simclr_feature, class_num)
    typicality = typicality.argsort()

    with torch.no_grad():
        for batch_idx, path in enumerate(testloader):
            s = str(path[0]) + "\n"
            with open('./batch_base.txt', 'a') as f: 
                f.write(s)

    with open('./batch_base.txt', 'r') as f: 
        losses = f.readlines()
    indices = []
    for i in losses :
        idx = int(i.split('/')[-1].split('.')[0])
        indices.append(idx)

    indices = np.array(indices)
    indices = indices.argsort()

    losses = np.array(losses)

    losses = losses[indices]

    print(losses[:10])

    losses = losses[typicality]

    for t in losses:
        with open('./batch_Atypicality.txt', 'a') as f: 
            f.write(str(t))
