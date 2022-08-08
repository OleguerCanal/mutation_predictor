import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader


class MutationsDataset(Dataset):
    def __init__(self, split, dataset_path):
        # Store split
        assert split in ["train", "val", "test"]
        self.split = split
        
        # Read text
        data = pd.read_csv(dataset_path, header=0, sep="\t")

        total_length = data.shape[0]
        if split == "train":
            self.data = data.head(int(0.8*total_length))
        elif split == "val":
            self.data = data.head(-int(0.2*total_length))

        alphabet = ["A", "C", "G", "T"]
        self.mapping = dict((c, i) for i, c in enumerate(alphabet))

    def __len__(self):
        return self.data.shape[0]

    def _onehot(self, sequence):
        one_hot_seq = torch.nn.functional.one_hot(
            torch.tensor([self.mapping[letter] for letter in sequence.split()]),
            num_classes=len(self.mapping),
        )
        return one_hot_seq

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sequence = row["sequence"]
        sequence_onehot = self._onehot(sequence)
        vals = row.drop(['sequence', 'label']).to_numpy().astype(np.float32)
        values = torch.tensor(vals)
        values = values.repeat(41, 1)  # since we only have the features of the center one we repeat them
        features = torch.cat((sequence_onehot, values), 1)

        label = self.mapping[row["label"]]

        return features, label


class MutationsDatamodule(pl.LightningDataModule):
    def __init__(self, dataset_path, batch_size) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(MutationsDataset(split="train", dataset_path=self.dataset_path),
                          batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(MutationsDataset(split="val", dataset_path=self.dataset_path),
                          batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(MutationsDataset(split="val", dataset_path=self.dataset_path),
                          batch_size=self.batch_size, shuffle=True)


if __name__=="__main__":
    md = MutationsDataset(split="train", dataset_path="../data/data.csv")
    features, label = md.__getitem__(0)
    print(features.shape)
    print(label)