import sys
import os
import glob

import dask
import numpy as np
from dask import delayed
from PIL import Image


@delayed
def get_image(src: str):
    np_image = np.array([open_image(img) for img in glob.glob(f"{src}\\*B[456].TIF")])
    return np_image



def open_image(src: str):
    return np.array(Image.open(src))


@delayed
def calculate_AFAI(img):
    img = img * 0
    return img

def get_mask(src: str):
    return np.array(Image.open(src))


@delayed
def no_observation_class(AFAI, mask):
    return AFAI * mask

def normalize_image():
    pass

@delayed
def result():
    pass


def main(argv):
    path = argv
    image_list = [get_image(os.path.join(path, folder)) for folder in os.listdir(path)]

    AFAI_list = [calculate_AFAI(img) for img in image_list]

    dask.compute(AFAI_list)


if __name__ == '__main__':
    main(sys.argv[1])
