import os, glob

import numpy as np
import pandas as pd
from tqdm import tqdm


# Helper functions
def rle_encode(img):
    """ TBD
    
    Args:
        img (np.array): 
            - 1 indicating mask
            - 0 indicating background
    
    Returns: 
        run length as string formated

    ref.: https://www.kaggle.com/stainsby/fast-tested-rle
    Source: https://www.kaggle.com/code/yiheng/3d-solution-with-monai-infer
    """
    
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

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

def id2filename(case_id, list_of_img_paths):
    """
    Input:
    -   Given an id input e.g. "case123_day20_slice_0065"
    -   List of path to images with the structure <case>/<case_day>/<scans>
    Output:
    -   Path to the image corresponding to the id e.g. train/case123/case123_day20/scans/slice_0065_266_266_1.50_1.50.png'
    """
    case_dir, case_day, _, slice_number = case_id.split("_")
    filtered_imgs = [img for img in list_of_img_paths if case_dir + '_' + case_day in img]
    filtered_imgs = [img for img in filtered_imgs if 'slice_' + slice_number in img]

    test = len(filtered_imgs) == 1
    if not test:
        print("Test: 'len(filtered_imgs) == 1' failed")
        print(f"len(filtered_imgs): {len(filtered_imgs)}")
        print(filtered_imgs)
    assert test
    return filtered_imgs[0]


def create_metadata_table(competition_dataset_folder = "../../../Dataset/uw-madison-gi-tract-image-segmentation/"):
    """Creates a pandas.DataFrame containing all relevant metadata. Each row corresponds to an image in the dataset.
    
    Args:
    string competition_dataset_folder: Path to competition directory 'uw-madison-gi-tract-image-segmentation'.
    
    Returns:
    pandas.DataFrame: Table with metadata
    """
    dataframe_path = os.path.join(competition_dataset_folder, "train.csv")
    pivot_df = get_pivot_table(dataframe_path)

    img_paths = glob.glob( os.path.join(competition_dataset_folder, "train/*/*/scans/*.png"))
    b = [a.split(os.sep)[1:3] + a.split(os.sep)[-1].replace(".png", "").split("_") + [len(os.listdir(os.sep.join(a.split(os.sep)[:-1])))] for a in tqdm(img_paths)]
    df_more_data = pd.DataFrame([["_".join(a[1:4])] + a[4:] for a in b], columns=["id", "sliceHeight", "sliceWidth", "pixelSpacingHeight", "pixelSpacingWidth", "num_slices"])

    big_df = pivot_df.merge(df_more_data, on="id")
    big_df[["sliceHeight", "sliceWidth", "num_slices"]] = big_df[["sliceHeight", "sliceWidth", "num_slices"]].astype(int)
    big_df[["pixelSpacingHeight", "pixelSpacingWidth"]] = big_df[["pixelSpacingHeight", "pixelSpacingWidth"]].astype(float)
    big_df["case"] = big_df["id"].str.split("_").apply(lambda x: x[0])
    big_df["case_day"] = big_df["id"].str.split("_").apply(lambda x: "_".join(x[:2]))

    df_paths = pd.DataFrame(
    [[
        "_".join([img_path.split(os.sep)[-3], "_".join(img_path.split(os.sep)[-1].split("_")[:2])]), 
        img_path, 
        os.sep.join(img_path.split(os.sep)[:-1]), 
        os.sep.join(img_path.split(os.sep)[:-3])
    ] for img_path in img_paths], 
    columns = ["id", "img_path", "scan_dir_path", "case_path"]
)
    big_df = big_df.merge(df_paths, on="id")
    return big_df