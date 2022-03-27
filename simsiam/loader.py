# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import random

from PIL import ImageFilter, Image
from torch.utils.data import Dataset

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class SupervisedSimSiamDataset(Dataset):
    def __init__(self, dataset):
        super(SupervisedSimSiamDataset, self).__init__()
        self.dset = dataset
        self.transform = self.dset.transform

        labels = np.array(self.dset.targets)
        self.unique_labels = np.unique(labels)
        self.indices = {
            label: np.where(self.dset.targets == label)[0] \
            for label in self.unique_labels}

    def __getitem__(self, index):
        x1 = self.dset.data[index]
        y = self.dset.targets[index]

        ind = np.random.choice(self.indices[y], 2, replace=False) # 2 in case selected the same index
        another_index = np.where(ind != index)[0][0] # [0, 1] if no overlap
        another_index = ind[another_index] # ind[0] if no overlap
        x2 = self.dset.data[another_index]

        if self.transform is not None:
            x1, x2 = Image.fromarray(x1), Image.fromarray(x2)
            x1, x2 = self.transform(x1), self.transform(x2)
        return [x1, x2], y

    def  __len__(self):
        return len(self.dset)
