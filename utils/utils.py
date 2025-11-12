import os
import random
from crip.io import *
import torch
import numpy as np
from collections import OrderedDict
from scipy import ndimage
import cv2
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from utils.pytools import read_raw_data




def get_random_RealDataSlices(VolLDCT, slice_num, slice_step, batch_size):
    _, H, W = VolLDCT.shape
    index_list = list(range(0, np.size(VolLDCT, 0)-slice_num, slice_step))
    if index_list[-1] + slice_num < np.size(VolLDCT, 0):
        index_list.append(np.size(VolLDCT, 0)-slice_num)
    LDCT_Batchs_Img = torch.zeros([batch_size, slice_num, H, W])
    for i in range(batch_size):
        index = random.choice(index_list)
        LDCTImg = VolLDCT[index:index + slice_num, :, :] * 100
        LDCT_Batchs_Img[i] = torch.FloatTensor(LDCTImg)
    return LDCT_Batchs_Img



def load_model(net, checkpoint):
    net.load_state_dict(checkpoint['state_dict'])
    # if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    #     net = torch.nn.DataParallel(net).cuda()
    # elif torch.cuda.is_available() and torch.cuda.device_count() == 1:
    #     net = net.cuda()
    return net


def save_model(net, optimizer, epoch, save_dir, scheduler=None):
    '''save model'''

    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)

    if 'module' in dir(net):
        state_dict = net.module.state_dict()
    else:
        state_dict = net.state_dict()

    if scheduler is None:
        torch.save({
            'state_dict': state_dict,
            'optimizer_state_dict': optimizer.state_dict()},
            os.path.join(save_dir, 'model_at_epoch_%03d.dat' % (epoch)))

    else:
        torch.save({
            'state_dict': state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler': scheduler.state_dict()},
            os.path.join(save_dir, 'model_at_epoch_%03d.dat' % (epoch)))

    print(os.path.join(save_dir, 'model_at_epoch_%03d.dat' % (epoch)))


def find_lastest_file(file_dir):
    lists = os.listdir(file_dir)
    lists.sort(key=lambda x: os.path.getmtime((file_dir + x)))
    file_latest = os.path.join(file_dir, lists[-1])

    return file_latest


def normalization(tensor):
    return (tensor - torch.min(tensor)) / (torch.max(tensor) - torch.min(tensor))


def gaussian_smooth(input, kernel_size=9, sigma=3.5):
    # inputs: batch, channel, width, height

    filter = np.float32(
        np.multiply(cv2.getGaussianKernel(kernel_size, sigma), np.transpose(cv2.getGaussianKernel(kernel_size, sigma))))
    filter = filter[np.newaxis, np.newaxis, ...]
    kernel = torch.FloatTensor(filter).cuda(input.get_device())
    if kernel.shape[1] != input.shape[1]:
        kernel = kernel.repeat(input.shape[1], 1, 1, 1)
    low = F.conv2d(input, kernel, padding=(kernel_size - 1) // 2, groups=input.shape[1])
    high = input - low

    return torch.cat([input, high], 1)


def multi_frequency(input, kernel_size=5, sigma=3.5):
    # inputs: batch, channel, width, height

    filter = np.float32(
        np.multiply(cv2.getGaussianKernel(kernel_size, sigma), np.transpose(cv2.getGaussianKernel(kernel_size, sigma))))
    filter = filter[np.newaxis, np.newaxis, ...]
    kernel = torch.FloatTensor(filter).cuda(input.get_device())
    if kernel.shape[1] != input.shape[1]:
        kernel = kernel.repeat(input.shape[1], 1, 1, 1)
    low = F.conv2d(input, kernel, padding=(kernel_size - 1) // 2, groups=input.shape[1])
    high = input - low

    return torch.cat([input, high, low], 1)


def get_edges(input, kernel_size=5, sigma=3.5):
    # inputs: batch, channel, width, height

    filter = np.float32(
        np.multiply(cv2.getGaussianKernel(kernel_size, sigma), np.transpose(cv2.getGaussianKernel(kernel_size, sigma))))
    filter = filter[np.newaxis, np.newaxis, ...]
    kernel = torch.FloatTensor(filter).cuda(input.get_device())
    if kernel.shape[1] != input.shape[1]:
        kernel = kernel.repeat(input.shape[1], 1, 1, 1)
    low = F.conv2d(input, kernel, padding=(kernel_size - 1) // 2, groups=input.shape[1])
    high = input - low

    return high


def x_derivative(input):
    # inputs: batch, channel, width, height

    fiter_first = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=np.float32)
    fiter_first = fiter_first[np.newaxis, np.newaxis, ...]
    fiter_kernel = torch.FloatTensor(fiter_first).cuda(input.get_device())
    derivative_first = F.conv2d(input, fiter_kernel, padding=(3 - 1) // 2)

    # fiter_second = np.array([[1, 2, 1],
    #                          [0, 0, 0],
    #                          [-1, -2, -1]], dtype=np.float32)
    # fiter_second = fiter_second[np.newaxis, np.newaxis, ...]
    # fiter_second = torch.FloatTensor(fiter_second).cuda(input.get_device())
    derivative_second = F.conv2d(derivative_first, fiter_kernel, padding=(3 - 1) // 2)

    return torch.cat([input, derivative_first, derivative_second], 1)


def cartesian2polar(image, num_angle):
    N = image.shape[0]
    M = image.shape[-1]
    theta = torch.linspace(np.deg2rad(0), np.deg2rad(179), num_angle)  # 1 x 1080
    # theta = torch.linspace(np.deg2rad(0), np.deg2rad(359), num_angle)  # 1 x 1080 # 12.25d
    # theta = torch.arange(np.deg2rad(-270), np.deg2rad(270), np.deg2rad(0.5))
    r = torch.linspace(0, M - 1, M) + 1 - (M + 1) / 2    # 1 x M
    # r = torch.arange(-M * 2.0, 2.0 * M , 1)
    # r = torch.arange(-0.8*M, 0.8*M + 1, 1)
    theta, r = torch.meshgrid(theta, r)  # 512 x M
    z = torch.polar(r, theta)
    X, Y = z.real.to(image.device), z.imag.to(image.device)
    grid = torch.stack((X / torch.max(X), Y / torch.max(Y)), 2).unsqueeze(0)  # (1, nv, nu, 2)  <= (N, H, W, 2)
    grid = grid.repeat(N, 1, 1, 1)
    return F.grid_sample(image, grid, align_corners=False)


def polar2cartesian(image):
    # N = image.shape[0]
    N, C, H, W = image.shape # 12.25
    # image = image[:, :, :H//2, :] # 12.25
    image_left, image_right = image.clone(), image.clone()
    image_left[..., :image.shape[-1] // 2] = image[..., :image.shape[-1] // 2]
    image_right[..., image.shape[-1] // 2:] = image[..., image.shape[-1] // 2:]
    image_left2right = torch.flip(image_left, dims=[-1])
    image_cat = torch.cat((image_right, image_left2right), dim=-2)

    M = image.shape[-1]
    X, Y = torch.arange(M).to(image.device), torch.arange(M).to(image.device)
    # X, Y = torch.meshgrid(X, Y, indexing='ij')
    X, Y = torch.meshgrid(X, Y)
    X, Y = X - (M - 1) / 2, Y - (M - 1) / 2
    r = torch.sqrt(X ** 2 + Y ** 2)
    theta = torch.atan2(Y, X)

    grid = torch.stack((r / (M // 2), theta / np.pi), 2).unsqueeze(0)  # (1, nv, nu, 2)  <= (N, H, W, 2)
    grid = grid.repeat(N, 1, 1, 1)
    img = F.grid_sample(image_cat, grid, align_corners=False)

    return torch.flip((torch.rot90(img, dims=[-2, -1], k=1)), dims=[-1])


def extract_patches_online(tensor, num=2):
    if tensor.ndim == 5:

        split_w = torch.chunk(tensor, chunks=num, dim=3)
        stack_w = torch.reshape(torch.stack(split_w, dim=0),
                                [num * tensor.shape[0], tensor.shape[1],
                                 tensor.shape[2], tensor.shape[3] // num, tensor.shape[4]])
        split_h = torch.chunk(stack_w, chunks=num, dim=4)
        stack_h = torch.reshape(torch.stack(split_h, dim=0),
                                [num * num * tensor.shape[0], tensor.shape[1],
                                 tensor.shape[2], tensor.shape[3] // num, tensor.shape[4] // num])

        return stack_h

    elif tensor.ndim == 4:

        split_w = torch.chunk(tensor, chunks=num, dim=2)
        # print('split_w', split_w.size())
        stack_w = torch.reshape(torch.stack(split_w, dim=0),
                                [num * tensor.shape[0], tensor.shape[1],
                                 tensor.shape[2] // num, tensor.shape[3]])
        # print('stack_w', stack_w.size())
        split_h = torch.chunk(stack_w, chunks=num, dim=3)
        # print('split_h', split_h.size())
        stack_h = torch.reshape(torch.stack(split_h, dim=0),
                                [num * num * tensor.shape[0], tensor.shape[1],
                                 tensor.shape[2] // num, tensor.shape[3] // num])
        # print('stack_h', stack_h.size())

        return stack_h

    else:
        print('Expect for the tensor with dim==5 or 4, other cases are not yet implemented.')


### mix two images
class MixUp_AUG:
    def __init__(self):
        self.dist = torch.distributions.beta.Beta(torch.tensor([1.2]), torch.tensor([1.2]))

    def aug(self, rgb_gt, rgb_noisy):
        bs = rgb_gt.size(0)
        indices = torch.randperm(bs)
        rgb_gt2 = rgb_gt[indices]
        rgb_noisy2 = rgb_noisy[indices]

        lam = self.dist.rsample((bs, 1)).view(-1, 1, 1, 1).cuda()

        rgb_gt = lam * rgb_gt + (1 - lam) * rgb_gt2
        rgb_noisy = lam * rgb_noisy + (1 - lam) * rgb_noisy2

        return rgb_gt, rgb_noisy


### mix two images
class MixUp_AUG_PI:
    def __init__(self):
        self.dist = torch.distributions.beta.Beta(torch.tensor([1.2]), torch.tensor([1.2]))

    def aug(self, rgb_gt, rgb_noisy, prior):
        bs = rgb_gt.size(0)
        indices = torch.randperm(bs)
        rgb_gt2 = rgb_gt[indices]
        rgb_noisy2 = rgb_noisy[indices]
        prior2 = prior[indices]

        lam = self.dist.rsample((bs, 1)).view(-1, 1, 1, 1).cuda()

        rgb_gt = lam * rgb_gt + (1 - lam) * rgb_gt2
        rgb_noisy = lam * rgb_noisy + (1 - lam) * rgb_noisy2
        prior = lam * prior + (1 - lam) * prior2

        return rgb_gt, rgb_noisy, prior


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
