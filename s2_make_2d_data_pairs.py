import os
import numpy as np
from utils.pytools import *
import time
import os

def make_data_list_txt(txt_file, data_list, ldct_dir='/mnt/RAR/Data/VolRAFDK/', rdct_dir='/mnt/RAR/Data/VolRFFDK/', sava_dir='/mnt/RAR/Data/TrainNPZ2D/'):

    make_dirs(sava_dir)

    for data_name in data_list:
        
        ldct_vol = np.load(ldct_dir + data_name)
        rdct_vol = np.load(rdct_dir + data_name)

        for index_z in range(np.size(ldct_vol, 0)):

            ldct_slice = ldct_vol[index_z, :, :]
            rdct_slice = rdct_vol[index_z, :, :]

            np.savez(sava_dir + data_name[:-4] + '_slice' + str(index_z+1), input=ldct_slice, label=rdct_slice)

            with open(txt_file, 'a') as f:
                f.write(sava_dir + data_name[:-4] + '_slice' + str(index_z+1) + '.npz\n')
            f.close()

def make_data_with_pi_list_txt(txt_file, data_list, ldct_dir='/mnt/RAR/Data/VolRAFDK/', pict_dir='/mnt/RAR/Data/VolRAFDK/', rdct_dir='/mnt/RAR/Data/VolRFFDK/', sava_dir='/mnt/RAR/Data/TrainNPZ2D/'):
    
    make_dirs(sava_dir)

    for data_name in data_list:
        
        ldct_vol = np.load(ldct_dir + data_name)
        pict_vol = np.load(pict_dir + data_name)
        rdct_vol = np.load(rdct_dir + data_name)

        for index_z in range(np.size(ldct_vol, 0)):

            ldct_slice = ldct_vol[index_z, :, :]
            pict_slice = pict_vol[index_z, :, :]
            rdct_slice = rdct_vol[index_z, :, :]

            np.savez(sava_dir + data_name[:-4] + '_slice' + str(index_z+1), input=ldct_slice, prior=pict_slice, label=rdct_slice)

            with open(txt_file, 'a') as f:
                f.write(sava_dir + data_name[:-4] + '_slice' + str(index_z+1) + '.npz\n')
            f.close()

def make_p3d_data_list_txt(txt_file, data_list, slice_num=5, slice_step=2, ldct_dir='/mnt/RAR/Data/VolRAFDK/', rdct_dir='/mnt/RAR/Data/VolRFFDK/', sava_dir='/mnt/RAR/Data/TrainNPZ2D/'):

    make_dirs(sava_dir)

    for data_name in data_list:
        
        ldct_vol = np.load(ldct_dir + data_name)
        rdct_vol = np.load(rdct_dir + data_name)

        index_list = list(range(0, np.size(rdct_vol, 0)-slice_num, slice_step))
        if index_list[-1] + slice_num < np.size(rdct_vol, 0):
            index_list.append(np.size(rdct_vol, 0)-slice_num)

        # for index_z in range(0, np.size(ldct_vol, 0)-slice_num, slice_step):
        for index_z in index_list:

            ldct_slice = ldct_vol[index_z:index_z+slice_num, :, :]
            rdct_slice = rdct_vol[index_z:index_z+slice_num, :, :]

            np.savez(sava_dir + data_name[:-4] + '_slice' + str(index_z+1), input=ldct_slice, label=rdct_slice)

            with open(txt_file, 'a') as f:
                f.write(sava_dir + data_name[:-4] + '_slice' + str(index_z+1) + '.npz\n')
            f.close()

if __name__ == "__main__":

    txt_save_dir = './txt/'
    make_dirs(txt_save_dir)
    
    train_txt = txt_save_dir + 'train_img_p3d_c3.txt'
    valid_txt = txt_save_dir + 'valid_img_p3d_c3.txt'

    vol_list = os.listdir('../Data/VolRAFDK/')
    vol_list.sort()

    train_valid_patient_list = [item for item in vol_list if 'D' not in item]
    train_patient_list = train_valid_patient_list[:132]
    valid_patient_list = train_valid_patient_list[132:]

    test_patient_list = [item for item in vol_list if 'D' in item]

    if os.path.exists(train_txt) == False:
        make_p3d_data_list_txt(train_txt, train_patient_list, slice_num=3, slice_step=1, sava_dir='/mnt/RAR/Data/TrainNPZP3DC5/')

    import random
    with open(train_txt, 'r') as infile:
        lines = infile.readlines()
    random.shuffle(lines)
    with open(train_txt, 'w') as outfile:
        outfile.writelines(lines)

    if os.path.exists(valid_txt) == False:
        make_p3d_data_list_txt(valid_txt, valid_patient_list, slice_num=3, slice_step=1, sava_dir='/mnt/RAR/Data/TrainNPZP3DC5/')

    with open(valid_txt, 'r') as infile:
        lines = infile.readlines()
    random.shuffle(lines)
    with open(valid_txt, 'w') as outfile:
        outfile.writelines(lines)