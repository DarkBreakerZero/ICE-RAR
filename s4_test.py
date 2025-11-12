import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import os
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p
from models.models import *
from utils.utils import *
from utils.pytools import *
import pandas as pd
from crip.io import *
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

save_simu_path = r"/mnt/no1/RAR/Data/results"
# save_real_path = win_to_ubuntu_path(r"F:\Code_Python\ICE-RAR-PMB-Exp\RealRatE")
maybe_mkdir_p(save_simu_path)
# maybe_mkdir_p(save_real_path)

model_name = 'ICE-RAR'

model_path = r"/mnt/no1/ResUNetP3D/model_at_epoch_100.dat"
checkpoint = torch.load(model_path)
model = ResUNetP3D(in_chl=3, out_chl=3, model_chl=40)
model = load_model(model, checkpoint).to(device)
slice_num, slice_step, slice_vol = 3, 1, 80

weight_matrix = np.ones([slice_vol, 1, 1], dtype=np.float32) * slice_num
weight_matrix[:slice_num-1, :, :] = np.reshape(list(range(1, slice_num, slice_step)), [slice_num-1, 1, 1])
weight_matrix[1-slice_num:, :, :] = np.reshape(list(range(1, slice_num, slice_step))[::-1], [slice_num-1, 1, 1])

model.eval()

# simulation rat test(1-channel)
simu_rats_path = r"/mnt/no1/RAR/Data/Rat/simu_test/test_tifVolRAFDK"

simu_rats = sorted(os.listdir(simu_rats_path))
simu_rats = [file for file in simu_rats if not file.endswith('csv')]
for simu_rats_n in tqdm(simu_rats):

    # simu_rat = np.load(os.path.join(simu_rats_path, simu_rats_n)).astype(np.float32)
    # simu_rat = imreadRaw(os.path.join(simu_rats_path, simu_rats_n), 512, 512, np.float32, 80)
    simu_rat = imreadTiff(os.path.join(simu_rats_path, simu_rats_n))
    # simu_rat = imreadTiff(win_to_ubuntu_path(r"E:\dataset\Rat\DifferentBin\recon_RatE_lo.tif"))
    VolPredict = np.zeros_like(simu_rat)

    index_list = list(range(0, np.size(simu_rat, 0) - slice_num, slice_step))
    if index_list[-1] + slice_num < np.size(simu_rat, 0):
        index_list.append(np.size(simu_rat, 0) - slice_num)

    for index in index_list:

        LDCTImg = simu_rat[index: index+slice_num, :, :] * 100
        LDCTImg = LDCTImg[np.newaxis, ...]
        LDCTImg = torch.FloatTensor(LDCTImg).to(device)


        with torch.no_grad():
            predictImg, _ = model(LDCTImg)
            predictImg = np.squeeze(predictImg.data.cpu().numpy())
            VolPredict[index:index+slice_num, :, :] += predictImg / 100  # average

    VolPredict = VolPredict / weight_matrix

    imwriteTiff(VolPredict, os.path.join(save_simu_path, simu_rats_n.replace('.tif', '_ICE-RAR.tif')))

# real_test
_, VolLDCT = read_raw_data(r"/mnt/no1/RAR/Data/Rat/real_test/RatE.raw", 512, 512)

VolPredict = np.zeros_like(VolLDCT)

index_list = list(range(0, np.size(VolLDCT, 0)-slice_num, slice_step))
if index_list[-1] + slice_num < np.size(VolLDCT, 0):
    index_list.append(np.size(VolLDCT, 0)-slice_num)

for index in index_list:

    LDCTImg = VolLDCT[index:index+slice_num, :, :] * 100
    LDCTImg = LDCTImg[np.newaxis, ...]
    LDCTImg = torch.FloatTensor(LDCTImg).to(device)

    with torch.no_grad():

        predictImg, _ = model(LDCTImg)
        predictImg = np.squeeze(predictImg.data.cpu().numpy())

    VolPredict[index:index + slice_num, :, :] += predictImg / 100  #average
VolPredict = VolPredict / weight_matrix

imwriteTiff(VolPredict.astype(np.float32), f'{save_simu_path}/RatE_ICE-RAR.tif')