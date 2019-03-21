'''
This is for the MarioQA dataset, where we take Base64 Images and convert into a format recognizable by our NN

'''
import cv2
import os
import base64
from os import path
from os.path import join, isdir, isfile
import numpy as np
import h5py

IMAGE_DIMENSION = 128
WORKERS = 20
DATA_DIR = 'data/mario_raw/'
OUT_DATA_DIR = 'data/mario/'

def process(workingpath, filename):

    filepath = path.join(workingpath, filename)
    with open(filepath) as f:
        b64_img = f.read()

    nparr = np.fromstring(base64.b64decode(b64_img), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # Resize Image
    resized_img = cv2.resize(img,(IMAGE_DIMENSION, IMAGE_DIMENSION))

    resized_img = resized_img[..., ::-1] # Convert cv2 BGR to RGB
    resized_img_normalized = resized_img / 255

    return resized_img_normalized

def extract_id(filename):
    a = filename.split('_')[1] # _number.dat
    b = a.split('.')[0] # number
    return int(b) - 1 # 0-index


def main():


    files = {}
    for f in  os.listdir(DATA_DIR):
        subdir = join(DATA_DIR, f)
        if path.isdir(subdir):

            ls = []
            for dat in os.listdir(subdir):
                file_path = join(subdir, dat)
                if dat.endswith('.dat'):
                    id = extract_id(dat)
                    ls.append(dat)
            files[f] = ls

    with h5py.File(OUT_DATA_DIR + "playdata.hdf5", "w") as f:
        for dir, files in files.items():
            working_path = join(DATA_DIR, dir)
            num_files = len(files) # Last

            ds = f.create_dataset(dir, (num_files, IMAGE_DIMENSION, IMAGE_DIMENSION, 3), )
            print('>> Now in dir', dir)

            for idx, file in enumerate(files):
                if idx % 100 == 0 and idx > 0:
                    print('>>>> Processing file number', idx)

                id = extract_id(file)
                img = process(working_path, file)
                ds[id] = img

            print('>> Processing complete, writing to HDF5 file')













if __name__ == '__main__':
    main()