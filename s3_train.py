import os
from utils.read_2d_npz_img import npz_reader_2d
from utils.losses import *
from utils.utils import *
from utils.sophia import *
import scipy
import numpy as np
import random
import torch.nn
from torch.utils.data import DataLoader
from torch.backends import cudnn
import time
import torch.optim as optim
import argparse
from models.models import *
from utils import *
from datetime import datetime
from utils.pytools import *
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
# from models.perceptual_UNet import *
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def train(train_loader, model, loss_mse, optimizer, scheduler, writer, epoch):

    batch_time = AverageMeter()
    loss_mse_scalar = AverageMeter()
    loss_mse_roi_scalar = AverageMeter()

    model.train()
    end = time.time()

    step = 0

    for data in tqdm(train_loader):

        RDCTImg = data["rdct"]
        RDCTImg = RDCTImg.cuda()

        # print(RDCTImg.size())

        LDCTImg = data["ldct"]
        # LDCTImg = LDCTImg[:, 2:3, :, :]
        LDCTImg = LDCTImg.cuda()

        if epoch > 4:
            RDCTImg, LDCTImg = MixUp_AUG().aug(RDCTImg, LDCTImg)

        # print(LDCTImg.shape)
        # LDCTImg[LDCTImg < 0] = 0

        predictImg, predictImgROI = model(LDCTImg)


        loss1 = loss_mse(predictImg, RDCTImg)
        loss2 = loss_mse(predictImgROI, F.upsample(RDCTImg[:, :, 192:320, 192:320], size=(512, 512), mode='bilinear', align_corners=True))


        loss = loss1 + loss2

        loss_mse_scalar.update(loss1.item())
        loss_mse_roi_scalar.update(loss2.item())

        batch_time.update(time.time() - end)
        end = time.time()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step += 1

    writer.add_scalars('loss/mse', {'train_mse_loss': loss_mse_scalar.avg}, epoch + 1)
    writer.add_scalars('loss/mse_roi', {'train_mse_roi_loss': loss_mse_roi_scalar.avg}, epoch + 1)

    writer.add_image('train img/label-predict-input img', normalization(torch.cat([RDCTImg[0,1:2, :, :], predictImg[0,1:2, :, :], LDCTImg[0,1:2, :, :]], 2)), epoch + 1)
    writer.add_image('train img roi/label-predict-input roi', normalization(torch.cat([F.upsample(RDCTImg[:, :, 192:320, 192:320], size=(512, 512), mode='bilinear', align_corners=True)[0,1:2, :, :], predictImgROI[0,1:2, :, :], F.upsample(LDCTImg[:, :, 192:320, 192:320], size=(512, 512), mode='bilinear', align_corners=True)[0,1:2, :, :]], 2)), epoch + 1)

    writer.add_image('train img/residual img', normalization(torch.abs(RDCTImg[0,1:2, :, :] - predictImg[0,1:2, :, :])), epoch + 1)


    scheduler.step()
    print('Train Epoch: {}\t train_mse_loss: {:.6f}\t'.format(epoch + 1, loss_mse_scalar.avg))

def valid(valid_loader, model, loss_mse, loss_ssim, writer, epoch):

    batch_time = AverageMeter()
    loss_mse_scalar = AverageMeter()
    loss_ssim_scalar = AverageMeter()
    model.eval()
    end = time.time()

    for data in tqdm(valid_loader):

        RDCTImg = data["rdct"]
        # RDCTImg = RDCTImg[:, 2:3, :, :]
        RDCTImg = RDCTImg.cuda()

        LDCTImg = data["ldct"]
        # LDCTImg = LDCTImg[:, 2:3, :, :]
        LDCTImg = LDCTImg.cuda()

        with torch.no_grad():

            predictImg, _ = model(LDCTImg)
            loss1 = loss_mse(predictImg, RDCTImg)
            loss2 = loss_ssim(predictImg, RDCTImg)

        loss_mse_scalar.update(loss1.item())
        loss_ssim_scalar.update(loss2.item())
        batch_time.update(time.time() - end)
        end = time.time()

    writer.add_scalars('loss/mse', {'valid_mse_loss': loss_mse_scalar.avg}, epoch+1)
    writer.add_scalars('loss/ssim', {'valid_ssim_loss': loss_ssim_scalar.avg}, epoch+1)
    writer.add_image('validation img/label-predict-input img', normalization(torch.cat([RDCTImg[0,1:2, :, :], predictImg[0,1:2, :, :], LDCTImg[0,1:2, :, :]], 2)), epoch + 1)
    # writer.add_image('validation img/reference img', normalization(RDCTImg[0,1:2, :, :]), epoch + 1)
    # writer.add_image('validation img/predict img', normalization(predictImg[0,1:2, :, :]), epoch + 1)
    # writer.add_image('validation img/ldct img', normalization(LDCTImg[0,1:2, :, :]), epoch + 1)
    writer.add_image('validation img/residual img', normalization(torch.abs(RDCTImg[0,1:2, :, :] - predictImg[0,1:2, :, :])), epoch + 1)

    print('Valid Epoch: {}\t valid_mse_loss: {:.6f}\t'.format(epoch + 1, loss_mse_scalar.avg))

def test(model, writer, epoch):

    batch_time = AverageMeter()
    model.eval()
    end = time.time()

    _, VolLDCT = read_raw_data('/mnt/RAR/Data/RatPCDImage/RatE.raw', 512, 512)

    VolPredict = np.zeros_like(VolLDCT)

    slice_num, slice_step = 3, 3

    index_list = list(range(0, np.size(VolLDCT, 0)-slice_num, slice_step))
    if index_list[-1] + slice_num < np.size(VolLDCT, 0):
        index_list.append(np.size(VolLDCT, 0)-slice_num)

    for index in index_list:

        LDCTImg = VolLDCT[index:index+slice_num, :, :] * 100
        LDCTImg = LDCTImg[np.newaxis, ...]
        LDCTImg = torch.FloatTensor(LDCTImg).cuda()

        with torch.no_grad():

            predictImg, _ = model(LDCTImg)
            predictImg = np.squeeze(predictImg.data.cpu().numpy())

        VolPredict[index:index+slice_num, :, :] = predictImg

        batch_time.update(time.time() - end)
        end = time.time()

    z_index = np.random.randint(0, VolPredict.shape[0])
    LDCTImgTB = VolLDCT[z_index, :, :] * 100
    LDCTImgTB = LDCTImgTB[np.newaxis, np.newaxis, ...]
    LDCTImgTB = torch.FloatTensor(LDCTImgTB).cuda()

    predictImgTB = VolPredict[z_index, :, :]
    predictImgTB = predictImgTB[np.newaxis, np.newaxis, ...]
    predictImgTB = torch.FloatTensor(predictImgTB).cuda()

    writer.add_image('test img/predict-input img', normalization(torch.cat([LDCTImgTB[0,:, :, :], predictImgTB[0,:, :, :]], 2)), epoch + 1)


if __name__ == "__main__":
          

    cudnn.benchmark = True

    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    result_path = '/mnt/RAR/Data/runs/ResUNetP3DFAV4TrainABValC/logs/'
    save_dir = '/mnt/RAR/Data/runs/ResUNetP3DFAV4TrainABValC/checkpoints/'

    # Get dataset
    train_dataset = npz_reader_2d(paired_data_txt='/mnt/RAR/Data/txt/train_img_p3d_c3.txt')
    train_loader = DataLoader(train_dataset, batch_size=4, num_workers=64, shuffle=True)

    valid_dataset = npz_reader_2d(paired_data_txt='/mnt/RAR/Data/txt/valid_img_p3d_c3.txt')
    valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=8, shuffle=True)

    model = ResUNetP3D(in_chl=3, out_chl=3, model_chl=40)
    loss_mse = torch.nn.MSELoss()
    loss_ssim = SSIM()

    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.95)

    if os.path.exists(save_dir) is False:

        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

        model = model.cuda()

    else:
        checkpoint_latest = torch.load(find_lastest_file(save_dir))
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(load_model(model, checkpoint_latest)).cuda()
        else:
            model = load_model(model, checkpoint_latest).cuda()
        optimizer.load_state_dict(checkpoint_latest['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint_latest['lr_scheduler'])
        print('Latest checkpoint {0} loaded.'.format(find_lastest_file(save_dir)))

    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
    log_dir = os.path.join(result_path, time_str)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    print("*"*20 + "Start Train" + "*"*20)

    for epoch in range(0, 100):

        print("*" * 20 + "Epoch: " + str(epoch + 1).rjust(4, '0') + "*" * 20)

        train(train_loader, model, loss_mse, optimizer, scheduler, writer, epoch)
        valid(valid_loader, model, loss_mse, loss_ssim, writer, epoch)
        test(model, writer, epoch)

        save_model(model, optimizer, epoch + 1, save_dir, scheduler)

