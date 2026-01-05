import os
import json
import random
from PIL import Image
from abc import abstractmethod

import pickle
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
from torchvision import transforms


class DatasetBase(Dataset):
    def __init__(self,
                 data_root,
                 txt_file=None,
                 size=None,
                 lr_size=None,
                 interpolation="bicubic",
                 first_k=None
                 ):
        self.data_root = data_root
        if txt_file is not None:
            with open(txt_file, "r") as f:
                self.image_paths = f.read().splitlines()
        else:
            self.image_paths = sorted(os.listdir(data_root))
        if first_k is not None:
            self.image_paths = self.image_paths[:first_k]
        self._length = len(self.image_paths)
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, l)
                           for l in self.image_paths],
        }

        self.size = size
        self.lr_size = lr_size
        self.interpolation = {"linear": Image.LINEAR,
                              "bilinear": Image.BILINEAR,
                              "bicubic": Image.BICUBIC,
                              "lanczos": Image.LANCZOS,
                              }[interpolation]
    def __len__(self):
        return self._length

    def __getitem__(self, i):
        image_path = os.path.join(self.data_root, self.image_paths[i])
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)

        data = {}
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image_hr = np.array(image).astype(np.uint8)
        data["image_hr"] = (image_hr / 127.5 - 1.0).astype(np.float32)

        if self.lr_size is not None:
            lr_image = image.resize((self.lr_size, self.lr_size), resample=self.interpolation)
            lr_image = np.array(lr_image).astype(np.uint8)
            data["image_lr"] = (lr_image / 127.5 - 1.0).astype(np.float32)

        return data

class CTSRTrain(DatasetBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/train.txt", data_root="", **kwargs)


class CTSRValidation(DatasetBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/val.txt", data_root="", **kwargs)

