from sympy import divisors
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


def get_output_address(path):
    return path.split(".")[0] + ".npy"


def metafile_generator(path=''):
    # Create a meta file for nii files processing
    file_list = []
    for dirname, _, filenames in os.walk('liver_tumor_segmentation/dataset/' + path):
        for filename in filenames:
            file_list.append((dirname, filename))

    df_files = pd.DataFrame(file_list, columns=['dirname', 'filename'])
    df_files.sort_values(by=['filename'], ascending=True)

    # Map CT scan and mask
    df_files['mask_dirname'] = ''
    df_files['mask_filename'] = ''

    for i in range(len(df_files)):
        ct = f'volume-{i}.nii'
        mask = f'segmentation-{i}.nii'

        df_files.loc[df_files['filename'] == ct, 'mask_filename'] = mask
        df_files.loc[df_files['filename'] == ct, 'mask_dirname'] = 'liver_tumor_segmentation/segmentations/'

    # drop segment row
    df_files = df_files[df_files['mask_filename'] != ''].sort_values(by=['filename']).reset_index(drop=True)
    return df_files


def read_nii(path):
    data = nib.load(path)
    data_arr = data.get_fdata()
    # img_arr = np.squeeze(img_arr)
    image = np.stack([scan for scan in data_arr])
    print(f"Shape of dataset: {image.shape}")
    # np.save(get_output_address(path), image)
    return image


def sample_stack(stack, start_with=0, show_every=1):
    sample_number = stack.shape[2]
    cols = divisors(sample_number)[2]
    rows = round((sample_number - start_with) / (cols * show_every))
    print(f"(rows = {rows}, cols = {cols}, sample = {sample_number})")
    fig, axes = plt.subplots(rows, cols)
    for i in range(int((sample_number - start_with) / show_every)):
        # axes[i].imshow(slice.T, cmap="gray", origin="lower")
        ind = start_with + i * show_every
        axes[int(i / cols), int(i % cols)].set_title('slice %d' % ind)
        axes[int(i / cols), int(i % cols)].imshow(stack[:, :, ind], cmap='gray', origin="lower")
        axes[int(i / cols), int(i % cols)].axis('off')
    plt.show()


def show_histogram(image):
    plt.figure()
    plt.hist(image.flatten(), bins=50, color='c')
    # plt.xlim(0.1, 0.9)
    plt.ylim(0, 5000)
    plt.xlabel("Hounsfield Units (HU)")
    plt.ylabel("Frequency")
    plt.show()