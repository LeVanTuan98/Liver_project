import types
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

from function import *

# From https://radiopaedia.org/articles/windowing-ct
dicom_windows = types.SimpleNamespace(
    brain=(80, 40),
    subdural=(254, 100),
    stroke=(8, 32),
    brain_bone=(2800, 600),
    brain_soft=(375, 40),
    lungs=(1500, -600),
    mediastinum=(350, 50),
    abdomen_soft=(400, 50),
    liver=(150, 30),
    spine_soft=(250, 50),
    spine_bone=(1800, 400),
    custom=(200, 60)
)


# Scale pixel intensity by window width and window level
# l = window level or center aka brightness
# w = window width or range aka contrast
def windowed(data, w, l):
    px = data.copy()
    px_min = l - w//2 # //: chia lay phan nguyen
    px_max = l + w//2
    px[px < px_min] = px_min
    px[px > px_max] = px_max
    return (px-px_min)/(px_max-px_min)
    # return px

def plot_sample(array_list, color_map='nipy_spectral'):
    # Plots and a slice with all available annotations
    fig = plt.figure(figsize=(18, 15))

    plt.subplot(1, 4, 1)
    plt.imshow(array_list[0], cmap='bone')
    plt.title('Original Image')

    plt.subplot(1, 4, 2)
    plt.imshow(windowed(array_list[0].astype(np.float32), *dicom_windows.liver), cmap='bone')
    plt.title('Windowed Image')

    plt.subplot(1, 4, 3)
    plt.imshow(array_list[1])
    plt.title('Mask')

    plt.subplot(1, 4, 4)
    plt.imshow(array_list[0], cmap='bone')
    plt.imshow(array_list[1], alpha=0.5, cmap=color_map)
    plt.title('Liver & Mask')

    plt.show()


# df_files = metafile_generator(path='volume_pt1')
id_pat = 0
id_slice = 45
# ct_set = read_nii(f"{df_files.loc[id_pat, 'dirname']}/{df_files.loc[id_pat, 'filename']}")
# mask_set = read_nii(f"{df_files.loc[id_pat, 'mask_dirname']}/{df_files.loc[id_pat, 'mask_filename']}")
ct_set = read_nii("liver_tumor_segmentation/dataset/volume_pt1/volume-0.nii")
mask_set = read_nii("liver_tumor_segmentation/segmentations/segmentation-0.nii")

image = ct_set[..., id_slice]
mask = mask_set[..., id_slice]
windowed_image = windowed(image.astype(np.float32), *dicom_windows.liver)
sample = np.zeros_like(windowed_image)
sample[mask == 1] = windowed_image[mask == 1]
plot_sample([image, mask])

# fig = plt.figure(figsize=(18, 15))
#
# plt.subplot(1, 4, 1)
# plt.imshow(sample, cmap='bone')
# plt.title('Original Image')
#
# plt.subplot(1, 4, 2)
# plt.hist(windowed_image.flatten(), bins=50)
# plt.hist(sample.flatten(), bins=50)
# plt.ylim(0, 5000)
# plt.show()

########################################################################
row_size = windowed_image.shape[0]
col_size = windowed_image.shape[1]
#compute average value
mean = np.mean(windowed_image)
#compute standard deviation
std = np.std(windowed_image)

# Using Kmeans to seraprate foreground (soft tissue / bone) and background (lung/air) -> cluster=2
kmeans = KMeans(n_clusters=2).fit()










