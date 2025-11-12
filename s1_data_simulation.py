import tigre
from utils.tigre_demos import set_geometry
from tigre.utilities import CTnoise
import tigre.algorithms as algs
import os
from utils.pytools import *
from skimage import io
import cv2
import time
import tifffile
from scipy.ndimage import gaussian_filter, zoom

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

GeoMicroCT = set_geometry(DSD=200.0,
                          DSO=100.0,
                          nDetector=(64, 384),
                          dDetector=(0.2, 0.2),
                          nVoxel=(120, 512, 512),
                          dVoxel=(0.06, 0.06, 0.06),
                          nView=600)

RawRFVolList = os.listdir('../Data/RatEIDImageDA/') # Path of the augmented numerical phantoms.
RawRFVolList.sort()

sava_dir_ra = '../Data/VolRAFDK/'
make_dirs(sava_dir_ra)

sava_dir_rf = '../Data/VolRFFDK/'
make_dirs(sava_dir_rf)

sava_dir_ra_p = '../Data/ProjRAFDK/'
make_dirs(sava_dir_ra_p)

sava_dir_rf_p = '../Data/ProjRFFDK/'
make_dirs(sava_dir_rf_p)

dead_pixel_detector = tifffile.imread('./dead_pixel.tif') # dead pixels: 0, others: 1.

for index, file_name in enumerate(RawRFVolList):

    seams_detector = np.ones_like(dead_pixel_detector, dtype=np.float32)
    seam_factors = [random.uniform(0.9, 1.1) for _ in range(5)] # The ranges can be adjusted according to real data.
    for s in range(5):
        seams_detector[:, 64*(s+1)-1:64*(s+1)+1] = seam_factors[s] # The position can be adjusted according to real data.

    blocks_detector = np.ones_like(dead_pixel_detector, dtype=np.float32)
    blocks_factors = [random.uniform(0.95, 1.05) for _ in range(6)] # The ranges can be adjusted according to real data.
    for b in range(6):
        blocks_detector[:, 64*(s):64*(s+1)] = blocks_factors[b]

    tic = time.time()

    vol_data = np.load('../Data/RatEIDImageDA/' + file_name) # Path of the augmented numerical phantoms.

    scale_factor = np.random.uniform(0.92, 1.15)
    vol_data_zoomed = zoom(vol_data, scale_factor, order=3, mode='mirror') # The interpolation step can be improved by PyTorch interpolation.

    if np.size(vol_data, 0) > np.size(vol_data_zoomed, 0):

        vol_data_zoomed = np.concatenate([vol_data_zoomed[(np.size(vol_data, 0) - np.size(vol_data_zoomed, 0)):0:-1, :, :], vol_data_zoomed], 0)

    elif np.size(vol_data, 0) < np.size(vol_data_zoomed, 0):

        vol_data_zoomed = vol_data_zoomed[0:-(np.size(vol_data_zoomed, 0) - np.size(vol_data, 0)), :, :]

    vol_data_filled = np.zeros_like(vol_data)

    if np.size(vol_data, 1) > np.size(vol_data_zoomed, 1):

        start_y, start_x = (np.size(vol_data, 1) - np.size(vol_data_zoomed, 1)) // 2, (np.size(vol_data, 2) - np.size(vol_data_zoomed, 2)) // 2

        vol_data_filled[:, start_y:start_y+np.size(vol_data_zoomed, 1), start_x:start_x+np.size(vol_data_zoomed, 2)] = vol_data_zoomed

    elif np.size(vol_data, 1) < np.size(vol_data_zoomed, 1):

        start_y, start_x = (np.size(vol_data_zoomed, 1) - np.size(vol_data, 1)) // 2, (np.size(vol_data_zoomed, 2) - np.size(vol_data, 2)) // 2

        vol_data_filled = vol_data_zoomed[:, start_y:start_y+np.size(vol_data, 1), start_x:start_x+np.size(vol_data, 2)]

    else:

        vol_data_filled = vol_data_zoomed

    vol_data_padded = np.concatenate([vol_data_filled[20:0:-1, :, :], vol_data_filled, vol_data_filled[-2:-22:-1, :, :]], 0)

    proj_data = tigre.Ax(vol_data_padded, GeoMicroCT, GeoMicroCT.angles)
    #proj_data = proj_data * 0.7
    pixel_level_detector_response_2d = np.random.uniform(0.93, 1.07, size=[64, 384])
    
    for index_radom in range(512):

        x_index = np.random.randint(5, pixel_level_detector_response_2d.shape[0] - 5)
        y_index = np.random.randint(5, pixel_level_detector_response_2d.shape[1] - 5)
        response_range = np.random.randint(1, 6)

        if index_radom < 128:
            pixel_level_detector_response_2d[x_index:x_index+response_range, y_index] = gaussian_filter(pixel_level_detector_response_2d[x_index:x_index+response_range, y_index], sigma=1)
        else:
            pixel_level_detector_response_2d[x_index, y_index:y_index+response_range] =  gaussian_filter(pixel_level_detector_response_2d[x_index, y_index:y_index+response_range], sigma=1)

    pixel_level_detector_response_2d = pixel_level_detector_response_2d * dead_pixel_detector * seams_detector * blocks_detector

    detector_response_3d = np.float32(np.repeat(pixel_level_detector_response_2d[np.newaxis, ...], repeats=600, axis=0))

    noisy_proj_data = CTnoise.add(proj_data, Poisson=4.5e3, Gaussian=np.array([0, 10]))

    noisy_ring_proj_data = noisy_proj_data * detector_response_3d

    vol_fdk = algs.fdk(proj_data, GeoMicroCT, GeoMicroCT.angles, filter="ram_lak")
    noisy_vol_fdk = algs.fdk(noisy_ring_proj_data, GeoMicroCT, GeoMicroCT.angles, filter="ram_lak")
    vol_fdk = vol_fdk[20:-20, :, :]
    noisy_vol_fdk = noisy_vol_fdk[20:-20, :, :]

    np.save(f'{sava_dir_ra}{file_name}', noisy_vol_fdk)
    np.save(f'{sava_dir_rf}{file_name}', vol_fdk)
    np.save(f'{sava_dir_ra_p}{file_name}', noisy_ring_proj_data)
    np.save(f'{sava_dir_rf_p}{file_name}', proj_data)

    toc = time.time()

    print(f'Current file: {file_name}, vol fdk size: {np.shape(vol_fdk)}, finished {index+1}/{len(RawRFVolList)}, processing time: {int(toc-tic)} seconds.')