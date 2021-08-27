from function import *
import cv2 as cv
import numpy as np
import argparse
import random as rng
import types
rng.seed(12345)

from fastai.basics import *
from fastai.vision.all import *
from fastai.data.transforms import *

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
    # return (px - px_min)/(px_max - px_min)
    return px



ct_set = read_nii("liver_tumor_segmentation/dataset/volume_pt1/volume-0.nii")
image = ct_set[..., 61]
# Show source image
cv.imshow('Source Image', image)
print(type(image[0, 0]))



windowed_image = windowed(image.astype(np.float32), *dicom_windows.liver)
cv.imshow('Windowed Image', windowed_image)
# print(windowed_image)
print(type(windowed_image[0, 0]))





cv.waitKey()