# Use the dataset from
# https://github.com/deepmind/deepmind-research/blob/master/adversarial_robustness/iclrw2021doing/README.md

import torch
from torch.utils import data
import numpy as np

from .base import Dataset

__all__ = ["DDPM"]


# https://github.com/yaircarmon/semisup-adv/blob/master/dataloader.py
class BalancedSampler(data.Sampler):
    def __init__(
        self,
        labels,
        batch_size,
        balanced_fraction=0.5,
        num_batches=None,
        label_to_balance=-1,
    ):
        self.minority_inds = [
            i for (i, label) in enumerate(labels) if label != label_to_balance
        ]
        self.majority_inds = [
            i for (i, label) in enumerate(labels) if label == label_to_balance
        ]
        self.batch_size = batch_size
        balanced_batch_size = int(batch_size * balanced_fraction)
        self.minority_batch_size = batch_size - balanced_batch_size

        if num_batches is not None:
            self.num_batches = num_batches
        else:
            self.num_batches = int(
                np.ceil(len(self.minority_inds) / self.minority_batch_size)
            )

        super().__init__(labels)

    def __iter__(self):
        batch_counter = 0
        while batch_counter < self.num_batches:
            minority_inds_shuffled = [
                self.minority_inds[i] for i in torch.randperm(len(self.minority_inds))
            ]
            # Cycling through permutation of minority indices
            for sup_k in range(0, len(self.minority_inds), self.minority_batch_size):
                if batch_counter == self.num_batches:
                    break
                batch = minority_inds_shuffled[
                    sup_k : (sup_k + self.minority_batch_size)
                ]
                # Appending with random majority indices
                if self.minority_batch_size < self.batch_size:
                    batch.extend(
                        [
                            self.majority_inds[i]
                            for i in torch.randint(
                                high=len(self.majority_inds),
                                size=(self.batch_size - len(batch),),
                                dtype=torch.int64,
                            )
                        ]
                    )
                np.random.shuffle(batch)
                yield batch
                batch_counter += 1

    def __len__(self):
        return self.num_batches


class DDPM_Dataset:
    def __init__(self, npz_file, base_dataset):
        npzfile = np.load(npz_file)
        self.images = npzfile["image"]
        self.labels = npzfile["label"]

        self.base_dataset = base_dataset
        self.ddpm_index = [1] * self.labels.shape[0] + [0] * len(
            self.base_dataset._train_dataset
        )

    def __getitem__(self, index):
        if index < self.labels.shape[0]:
            img = self.images[index].transpose(2, 0, 1)
            img = torch.Tensor(img)
            label = self.labels[index]
            label = int(label)
        else:
            img, label = self.base_dataset._train_dataset[index - self.labels.shape[0]]
        return img, label

    def __len__(self):
        return self.labels.shape[0] + len(self.base_dataset._train_dataset)


class DDPM(Dataset):
    num_classes = -1
    dataset_name = "ddpm"

    def __init__(self, base_dataset, root, batch_size=128):
        self.num_classes = base_dataset.num_classes
        self.dataset_name = base_dataset.dataset_name + "_ddpm"
        npzfile = f"{root}/{base_dataset.dataset_name}_ddpm.npz"
        self.ddpm_dataset = DDPM_Dataset(npzfile, base_dataset)

        train_sampler = BalancedSampler(
            self.ddpm_dataset.ddpm_index,
            batch_size,
            balanced_fraction=0.5,
            label_to_balance=1,
        )

        self._train_loader = data.DataLoader(
            self.ddpm_dataset,
            batch_sampler=train_sampler,
            num_workers=1,
            pin_memory=True,
        )
        self._test_loader = data.DataLoader(
            base_dataset._test_dataset,
            batch_size=batch_size,
            num_workers=1,
            pin_memory=True,
            shuffle=False,
        )

    @property
    def train_loader(self):
        return self._train_loader

    @property
    def test_loader(self):
        return self._test_loader

    @property
    def val_loader(self):
        return None

    @property
    def mean(self):
        return torch.tensor((0.4914, 0.4822, 0.4465)).view(3, 1, 1).cuda()

    @property
    def std(self):
        return torch.tensor((0.2471, 0.2435, 0.2616)).view(3, 1, 1).cuda()
