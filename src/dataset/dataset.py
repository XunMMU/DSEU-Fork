import torch
from torch.utils.data import Dataset
import pandas as pd

class Hazeset(Dataset):
    def __init__(self, annotations_file: str):
        self.csv = pd.read_csv(annotations_file)

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index: int):
        #-> torch.Tensor:
        return "TODO"
