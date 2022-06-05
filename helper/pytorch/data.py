import os, glob

import cv2
import numpy as np
from torch.utils.data import Dataset

from ..helper import rle_decode, create_metadata_table

class UWMGI2022SegmentationTrainingDataset(Dataset):
    def __init__(self, dataset_path, transform=None):

        print("Preparing metadata dataframe...")
        self.df = create_metadata_table(dataset_path)
        self.transform = transform
        print("Metadata dataframe prepared.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, "img_path"]
        masks = self.df.loc[idx, ["large_bowel", "small_bowel", "stomach"]]
        masks.fillna("skip", inplace=True)
        
        
        image = cv2.imread(img_path, -1)
        
        assert image.dtype == np.uint16
        image = np.divide(image, 2**16) # Normalize [0, 1]
        
        mask = self._create_mask_array(masks, mask_shape=image.shape)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image, mask = transformed['image'], transformed['mask']

        return image, mask
    
    
    def _create_mask_array(self, masks, mask_shape):
        np_mask = list()

        for mask in masks:
            if mask == "skip":
                mask = np.zeros(mask_shape)
            else:
                mask = rle_decode(mask, shape=mask_shape)


            np_mask.append(mask)
        return np.stack(np_mask, axis=-1)

class UWMGI2022SegmentationInferenceDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.img_paths = glob.glob(os.path.join(dataset_path, "*/*/scans/*.png"))
        self.transform = transform
        print(f"Number of images found: {len(self.img_paths)}")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = cv2.imread(img_path, -1)
        
        assert image.dtype == np.uint16
        image = np.divide(image, 2**16) # Normalize [0, 1]
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']

        return image