import sys
import os
import glob

import dask
import numpy as np
from dask import delayed
from PIL import Image
import pyopencl as cl
import pyopencl.array as cl_array
import unpackqa
import time

from matplotlib import pyplot as plt

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags


class BufferPool:
    def __init__(self, ctx, size, count):
        self.context = ctx
        self.pool = [cl.Buffer(ctx, cl.mem_flags.READ_WRITE, size=size) for _ in range(count)]
        self.available = self.pool[:]

    def get_buffer(self):
        if not self.available:
            # If no available buffers, create a new one
            return cl.Buffer(self.context, cl.mem_flags.READ_WRITE, size=size)
        return self.available.pop()

    def release_buffer(self, buf):
        self.available.append(buf)
        # Optionally, release buffer explicitly (buf.release())

@delayed
def get_image(src: str):
    np_image = np.array([open_image(img) for img in glob.glob(f"{src}/*B[456].TIF")])
    return np_image


def open_image(src: str):
    return np.array(Image.open(src))


@delayed
def calculate_AFAI(img):
    (R, NIR, SWIR) = img[0].astype(np.float64), img[1].astype(np.float64), img[2].astype(np.float64)
    (lambda_R, lambda_NIR, lambda_SWIR) = (655, 865, 1610)
    lambda_ratio = (lambda_NIR - lambda_R) / (lambda_SWIR - lambda_R)

    # Allocating memory on GPU
    R_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=R)
    NIR_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=NIR)
    SWIR_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=SWIR)

    # Compile and execute kernel

    prg = cl.Program(ctx, """
    __kernel void afai(__global const double *R_g,
                       __global const double *NIR_g,
                       __global const double *SWIR_g,
                       __global double *AFAI_g,
                       const double ratio,
                       const int cols)
    {
        int gid = get_global_id(0) * cols + get_global_id(1);
        AFAI_g[gid] = NIR_g[gid] - (R_g[gid] + ((SWIR_g[gid] - R_g[gid]) * ratio));
        
    }
    """).build()

    AFAI_g = cl.Buffer(ctx, mf.WRITE_ONLY, R.nbytes)
    afai_knl = prg.afai
    afai_knl(queue, R.shape, None, R_g, NIR_g, SWIR_g, AFAI_g, np.float64(lambda_ratio), np.int32(R.shape[1]))

    # Copy results back
    AFAI = np.empty_like(R)
    cl.enqueue_copy(queue, AFAI, AFAI_g)

    return AFAI


def get_mask(src: str):
    mask_arr = open_image(glob.glob(f"{src}/*PIXEL.TIF")[0])
    mask = unpackqa.unpack_to_array(mask_arr,
                                    product='LANDSAT_8_C2_L2_QAPixel',
                                    flags=['Water'])
    return mask.astype('uint16')


@delayed
def no_observation_class(AFAI, mask):
    # Memory allocation
    mask = mask.astype(np.float64)
    AFAI_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=AFAI)
    mask_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=mask)

    prg = cl.Program(ctx, """
        __kernel void no_observation(__global const double *AFAI_g,
                                     __global const double *mask_g,
                                     __global double *result_g,
                                     const int cols)
        {
            int gid = get_global_id(0) * cols + get_global_id(1);
            result_g[gid] = AFAI_g[gid] * mask_g[gid];

        }
        """).build()
    result_g = cl.Buffer(ctx, mf.WRITE_ONLY, AFAI.nbytes)
    noc_knl = prg.no_observation
    noc_knl(queue, AFAI.shape, None, AFAI_g, mask_g, result_g, np.int32(AFAI.shape[1]))

    no_observation = np.empty_like(AFAI)
    cl.enqueue_copy(queue, no_observation, result_g)
    # Copy results back

    return no_observation

@delayed
def normalize_image(img):

    img_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=img)

    prg = cl.Program(ctx, """
            __kernel void normalize(__global const double *img_g,
                                    __global double *result_g,
                                    const int cols,
                                    const double min,
                                    const double max)
            {
                int gid = get_global_id(0) * cols + get_global_id(1);
                result_g[gid] = 1 - ((img_g[gid] - min) / (max - min));

            }
            """).build()
    result_g = cl.Buffer(ctx, mf.WRITE_ONLY, img.nbytes)
    norm_knl = prg.normalize
    norm_knl(queue, img.shape, None, img_g, result_g, np.int32(img.shape[1]), np.min(img), np.max(img))

    normalized_img = np.empty_like(img)
    cl.enqueue_copy(queue, normalized_img, result_g)

    return normalized_img


@delayed
def save_image(img, src):
    Image.fromarray(img).save(f"result/{src}_AFAI.TIF")



def main(argv):
    path = argv
    start_time = time.time()
    image_list = [get_image(os.path.join(path, folder)) for folder in os.listdir(path)]
    AFAI_list = [calculate_AFAI(img) for img in image_list]
    mask_list = [get_mask(os.path.join(path, folder)) for folder in os.listdir(path)]
    no_observation_list = [no_observation_class(AFAI, mask) for AFAI, mask in zip(AFAI_list, mask_list)]
    normalized_list = [normalize_image(img) for img in no_observation_list]
    save = [save_image(img, src) for img, src in zip(normalized_list, os.listdir(path))]
    dask.compute(*save)

    #dask.visualize(*save)
    end_time = time.time()
    print(f'GPU: {end_time - start_time}')





if __name__ == '__main__':
    main(sys.argv[1])
