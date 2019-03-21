'''
This is for the MarioQA dataset, where we take Base64 Images and convert into a format recognizable by our NN

'''
import cv2
import os
from os import path
from os.path import join, isdir, isfile
import numpy as np

IMAGE_DIMENSION = 128
WORKERS = 20
DATA_DIR = 'data/mario/'

def process(workingpath, filename):

    filepath = path.join(workingpath, filename)
    with open(filepath) as f:
        b64_img = f.read()

    nparr = np.fromstring(b64_img.decode('base64'), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # Resize Image
    resized_img = cv2.resize(img,(IMAGE_DIMENSION, IMAGE_DIMENSION))

def extract_id(filename):
    a = filename.split('_')[1] # _number.dat
    b = a.split('.')[0] # number
    return int(b)


def main():
    # Check if all of the listing are consequtive
    for f in  os.listdir(DATA_DIR):
        subdir = join(DATA_DIR, f)
        if path.isdir(subdir):
            print("entering dir %s" % subdir)

            ls = []

            for dat in os.listdir(DATA_DIR):
                file_path = join(subdir, dat)
                if dat.endswith('.dat'):
                    id = extract_id(dat)
                    ls.append(id)

            sorted_ls = sorted(ls)
            print('Total files', len(sorted_ls), 'largest number', sorted_ls[-1])
            print('')





if __name__ == '__main__':
    main()