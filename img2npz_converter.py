from os import listdir
from os.path import join, isfile
import cv2
import numpy as np
import matplotlib.pyplot as plt


def main():
    in_path = 'data/megaman'
    outpath = 'data/megaman.npz'
    img_files = [join(in_path, f) for f in listdir(in_path)
                        if f.endswith('.png')]


    imgs_BGR = np.array([cv2.imread(f) for f in img_files])

    imgs = imgs_BGR[..., ::-1] # Converting from Open CV BGR to RGB on the last dimension
    plt.imshow(imgs[0])

    print('hello_world')




    pass

if __name__ == '__main__':
    main()