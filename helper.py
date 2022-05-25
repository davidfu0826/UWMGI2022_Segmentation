import os 

import numpy as np
import pandas as pd


# Helper functions
def rle_decode(mask_rle, shape, color=1):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height, width, channels) of array to return 
    color: color for the mask
    Returns numpy array (mask)

    Source: https://www.kaggle.com/code/arunamenon/image-segmentation-eda-mask-r-cnn-train
    '''
    s = mask_rle.split()
    
    starts = list(map(lambda x: int(x) - 1, s[0::2]))
    lengths = list(map(int, s[1::2]))
    ends = [x + y for x, y in zip(starts, lengths)]
    
    if len(shape)==3:
        img = np.zeros((shape[0] * shape[1], shape[2]), dtype=np.float32)
    else:
        img = np.zeros(shape[0] * shape[1], dtype=np.float32)
         
    for start, end in zip(starts, ends):
        img[start : end] = color
    
    return img.reshape(shape)

def get_pivot_table(dataframe_path):
    """
    Obtain DataFrame needed by get_mask function
    
    """
    df = pd.read_csv(dataframe_path)
    return df.pivot(index="id", columns="class", values="segmentation").reset_index()

def _get_mask_row(img_path, pivot_df):
    case, case_day, _, filename = img_path.split(os.sep)[-4:]
    assert filename == os.path.basename(img_path)
    case_id = "_".join([case_day] + filename.split("_")[:2])
    return pivot_df[pivot_df["id"] == case_id]

def get_image_size_by_path(img_path):
    filename = os.path.basename(img_path)
    h, w = filename.split("_")[2:4]
    return int(h), int(w)

def get_mask(img_path, pivot_df, labels=["large_bowel", "small_bowel", "stomach"]):
    pivot_row = _get_mask_row(img_path, pivot_df)
    img_h, img_w = get_image_size_by_path(img_path)

    masks = list()
    for label in labels:
        if not pivot_row[label].isna().item():
            masks.append(rle_decode(pivot_row[label].item(), shape=(img_w, img_h)))
        else:
            masks.append(np.zeros((img_w, img_h)))
    return np.stack(masks, axis=-1)