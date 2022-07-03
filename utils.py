import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import scipy.stats as st
import csv


# load image metadata (Image_ID, true label, and target label)
def load_ground_truth(csv_filename):
    image_id_list = []
    label_ori_list = []
    label_tar_list = []

    with open(csv_filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            image_id_list.append(row['ImageId'])
            label_ori_list.append(int(row['TrueLabel']) - 1)
            label_tar_list.append(int(row['TargetClass']) - 1)

    return image_id_list, label_ori_list, label_tar_list


# define TI
def gkern(kernlen=15, nsig=3):
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel


# define DI
def DI(X_in):
    rnd = np.random.randint(299, 330, size=1)[0]
    h_rem = 330 - rnd
    w_rem = 330 - rnd
    pad_top = np.random.randint(0, h_rem, size=1)[0]
    pad_bottom = h_rem - pad_top
    pad_left = np.random.randint(0, w_rem, size=1)[0]
    pad_right = w_rem - pad_left

    c = np.random.rand(1)
    if c <= 0.7:
        X_in_pad = F.interpolate(X_in, size=(rnd, rnd))
        # a trivial bug of the original code
        # X_out = F.pad(X_in_pad, (pad_left, pad_top, pad_right, pad_bottom), mode='constant', value=0)
        X_out = F.pad(X_in_pad, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        return X_out, rnd, pad_top, pad_left, c
    else:
        return X_in, rnd, pad_top, pad_left, c


# DI with specified parameters
def DI_pa(X_in, rnd, pad_top, pad_left, c):
    # rnd = np.random.randint(299, 330, size=1)[0]
    h_rem = 330 - rnd
    w_rem = 330 - rnd
    # pad_top = np.random.randint(0, h_rem, size=1)[0]
    pad_bottom = h_rem - pad_top
    # pad_left = np.random.randint(0, w_rem, size=1)[0]
    pad_right = w_rem - pad_left

    # c = np.random.rand(1)
    if c <= 0.7:
        X_in_pad = F.interpolate(X_in, size=(rnd, rnd))
        # X_out = F.pad(X_in_pad, (pad_left, pad_top, pad_right, pad_bottom), mode='constant', value=0)
        X_out = F.pad(X_in_pad, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        return X_out
    else:
        return X_in


# define Po+Trip
def Poincare_dis(a, b):
    L2_a = torch.sum(torch.square(a), 1)
    L2_b = torch.sum(torch.square(b), 1)

    theta = 2 * torch.sum(torch.square(a - b), 1) / ((1 - L2_a) * (1 - L2_b))
    distance = torch.mean(torch.acosh(1.0 + theta))
    return distance


def Cos_dis(a, b):
    a_b = torch.abs(torch.sum(torch.multiply(a, b), 1))
    L2_a = torch.sum(torch.square(a), 1)
    L2_b = torch.sum(torch.square(b), 1)
    distance = torch.mean(a_b / torch.sqrt(L2_a * L2_b))
    return distance


def Cos_dis_sign(a, b):
    a = a.view(-1, 1)
    b = b.view(-1, 1)
    a_b = torch.sum(torch.multiply(a, b))
    L2_a = torch.sum(torch.square(a))
    L2_b = torch.sum(torch.square(b))
    distance = a_b / torch.sqrt(L2_a * L2_b)

    return distance


def projAtoB(a, b):
    a = a.squeeze()
    b = b.squeeze()

    temp1 = (torch.mm(a.view(1, -1), b.view(-1, 1))) / (1e-20 + torch.mm(b.view(1, -1), b.view(-1, 1)))
    temp1reshape = temp1.view(1, 1)
    a_proj = torch.mm(temp1reshape, b.view(1, -1))
    a_proj = a_proj.view(a.shape)

    a_orth = a - a_proj
    corr = Cos_dis_sign(a_orth, b)

    return a_orth.unsqueeze(0)


def projAtoB_batch(a, b):
    batchsize = a.shape[0]
    a_orth_batch = torch.zeros_like(a)
    for i in range(batchsize):
        a_orth = projAtoB(a[i], b[i])
        a_orth_batch[i] = a_orth.squeeze()

    return a_orth_batch
