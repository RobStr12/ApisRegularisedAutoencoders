import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import time


class ApisDataset(Dataset):

    def __init__(self, paper, device="cpu"):
        data = np.loadtxt(f"./data/{paper}/data/count_matrix_{paper}.csv", delimiter=",", dtype=np.float32, skiprows=1)
        data = data.T
        data = data[1:]

        start = time.time()
        self.data = [[read / sum(reads) for read in reads] for reads in tqdm(data, total=len(data))]
        self.data = np.array(self.data, dtype=np.float32)

        self.tensor = torch.tensor(self.data).to(device)
        end = time.time()

        print(f"{device} took {(end-start) // 60} minutes and {(end - start) % 60:.2f} seconds to load the dataset")

        self.device = device
        self.n_samples = self.tensor.shape[0]
        self.n_genes = self.tensor.shape[1]

    def __str__(self):
        return str(self.tensor)

    def __getitem__(self, index):
        return self.tensor[index]

    def __len__(self):
        return len(self.tensor)
