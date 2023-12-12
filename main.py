import sys
import os
import glob

import dask
import numpy as np
from dask import delayed
from PIL import Image
import pyopencl as cl
import unpackqa
import time
import warnings

# Ignoring opencl warnings
warnings.filterwarnings('ignore')
os.environ['PYOPENCL_CTX'] = '0'

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags


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
    # This buffer will be reusable between R and AFAI
    R_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=R)
    NIR_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=NIR)
    SWIR_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=SWIR)

    # List of buffers to reuse later
    buffers = [R_g, NIR_g]
    # Compile and execute kernel

    prg = cl.Program(ctx, """
    __kernel void afai(__global double *R_g,
                       __global const double *NIR_g,
                       __global const double *SWIR_g,
                       const double ratio,
                       const int cols)
    {
        int gid = get_global_id(0) * cols + get_global_id(1);
        R_g[gid] = NIR_g[gid] - (R_g[gid] + ((SWIR_g[gid] - R_g[gid]) * ratio));
        
    }
    """).build(options=['-cl-single-precision-constant', '-cl-mad-enable'])

    afai_knl = prg.afai
    afai_knl.set_args(R_g, NIR_g, SWIR_g, np.float64(lambda_ratio), np.int32(R.shape[1]))
    cl.enqueue_nd_range_kernel(queue, afai_knl, R.shape, None)

    SWIR_g.release()

    return buffers, R.shape


@delayed
def get_mask(src: str):
    mask_arr = open_image(glob.glob(f"{src}/*PIXEL.TIF")[0])
    mask = unpackqa.unpack_to_array(mask_arr,
                                    product='LANDSAT_8_C2_L2_QAPixel',
                                    flags=['Water'])
    return mask.astype(np.float64)


@delayed
def no_observation_class(buffers, mask):
    # Memory allocation
    buffers, size = buffers
    AFAI = buffers[0]
    mask_g = buffers[1]
    cl.enqueue_copy(queue, mask_g, mask).wait()

    prg = cl.Program(ctx, """
        __kernel void no_observation(__global double *AFAI,
                                     __global const double *mask_g,
                                     const int cols)
        {
            int gid = get_global_id(0) * cols + get_global_id(1);
            AFAI[gid] = AFAI[gid] * mask_g[gid];

        }
        """).build(options=['-cl-single-precision-constant', '-cl-mad-enable'])
    # result_g = cl.Buffer(ctx, mf.WRITE_ONLY, AFAI.nbytes)
    noc_knl = prg.no_observation
    noc_knl.set_args(AFAI, mask_g, np.int32(size[1]))
    cl.enqueue_nd_range_kernel(queue, noc_knl, size, None)

    no_observation = np.empty_like(mask)
    cl.enqueue_copy(queue, no_observation, AFAI)

    # AFAI.release()
    mask_g.release()
    return no_observation, AFAI


@delayed
def normalize_image(img):
    img, buffer = img
    cl.enqueue_copy(queue, buffer, img).wait()
    prg = cl.Program(ctx, """
            __kernel void normalize(__global double *img_g,
                                    const int cols,
                                    const double min,
                                    const double max)
            {
                int gid = get_global_id(0) * cols + get_global_id(1);
                img_g[gid] = 1 - ((img_g[gid] - min)  / (max - min));

            }
            """).build(options=['-cl-single-precision-constant', '-cl-mad-enable'])
    norm_knl = prg.normalize
    norm_knl.set_args(buffer, np.int32(img.shape[1]), np.min(img), np.max(img))
    cl.enqueue_nd_range_kernel(queue, norm_knl, img.shape, None)

    normalized_img = np.empty_like(img)

    cl.enqueue_copy(queue, normalized_img, buffer)
    buffer.release()
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
    no_observation_list = [no_observation_class(buffers, mask) for buffers, mask in zip(AFAI_list, mask_list)]
    normalized_list = [normalize_image(img) for img in no_observation_list]
    save = [save_image(img, src) for img, src in zip(normalized_list, os.listdir(path))]

    dask.compute(*save)
    end_time = time.time()
    print(f'GPU: {end_time - start_time}')


if __name__ == '__main__':
    dask.config.set(scheduler='threads')
    main(sys.argv[1])
