import cv2
from tifffile import imread
from tqdm import tqdm
import glob
import h5py
import numpy as np

# %%
layers = glob.glob("/data/scroll_data/dl.ash2txt.org/full-scrolls/Scroll2.volpkg/volumes/20230210143520/*.tif")

# %%
for layer in tqdm(layers):
    layer_img = cv2.convertScaleAbs(imread(layer), alpha=(255.0/65535.0))
    break

# %%
with h5py.File('/data/scroll2.hdf5', "w") as f:
    dset = f.create_dataset("scan_volume", shape=(layer_img.shape[0],layer_img.shape[1], len(layers)), dtype=np.uint8, chunks=(256, 256, 256), shuffle=True, compression="lzf")
    # dset = f["scan_volume"]
    for layer_range in tqdm(range(0, len(layers), 256)):
        array_container = []
        for i in tqdm(range(layer_range, layer_range+256)):
            try:
                array_container.append(cv2.convertScaleAbs(imread(layers[i]), alpha=(255.0/65535.0)))
            except:
                print("layer broke on ", i)
                break
        array_container = np.stack(array_container, axis = -1)
        print(layer_range, layer_range+256)
        dset[:, :, layer_range:layer_range+256] = array_container