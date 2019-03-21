from os import listdir
from os.path import join, isfile
import cv2
import h5py
import numpy as np
import matplotlib.pyplot as plt


def main2():
    in_path = 'data/megaman'
    outpath = 'data/megaman.npz'
    img_files = [join(in_path, f) for f in listdir(in_path)
                        if f.endswith('.png')]


    imgs_BGR = np.array([cv2.imread(f) for f in img_files])

    imgs = imgs_BGR[..., ::-1] # Converting from Open CV BGR to RGB on the last dimension
    plt.imshow(imgs[0])

    print('hello_world')

def main():
    dataset_zip = np.load('data/megaman.npz')
    imgs   = dataset_zip['arr_0']
    print('writing and compressing lzf')
    with h5py.File('data/megaman.hdf5', "w") as f:
        f.create_dataset('megaman', data = imgs, compression='lzf')


if __name__ == '__main__':
    main()