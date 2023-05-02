"""
The proposed method in the single-model scenario.
For completeness, the codes of the compared methods (CE, Po+Trip, logits) are also included.
Our codes are heavily borrowed from:
https://github.com/ZhengyuZhao/Targeted-Tansfer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from PIL import Image
import numpy as np
from utils import gkern, DI, DI_pa, Poincare_dis, Cos_dis, load_ground_truth, Cos_dis_sign
from utils import projAtoB_batch
import scipy.io as scio
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# simple Module to normalize an image
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)

    def forward(self, x):
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]


channels = 3
kernel_size = 5
kernel = gkern(kernel_size, 3).astype(np.float32)
gaussian_kernel = np.stack([kernel, kernel, kernel])
gaussian_kernel = np.expand_dims(gaussian_kernel, 1)
gaussian_kernel = torch.from_numpy(gaussian_kernel).cuda()

# 1. Model: load the pretrained models
# source model: model_2; target models: model_1, 3, 4
model_1 = models.inception_v3(pretrained=True, transform_input=True).eval()
model_2 = models.resnet50(pretrained=True).eval()
model_3 = models.densenet121(pretrained=True).eval()
model_4 = models.vgg16_bn(pretrained=True).eval()

for param in model_1.parameters():
    param.requires_grad = False
for param in model_2.parameters():
    param.requires_grad = False
for param in model_3.parameters():
    param.requires_grad = False
for param in model_4.parameters():
    param.requires_grad = False

device = torch.device("cuda:0")
model_1.to(device)
model_2.to(device)
model_3.to(device)
model_4.to(device)

torch.backends.cudnn.deterministic = True

# 2. Data: 1000 images from the ImageNet-Compatible dataset
# values are standard normalization for ImageNet images,
# from https://github.com/pytorch/examples/blob/master/imagenet/main.py
norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
trn = transforms.Compose([transforms.ToTensor(), ])
image_id_list, label_ori_list, label_tar_list = load_ground_truth('./dataset/images.csv')

# 3. Parameters
batch_size = 20
beta = 0.2
beta2 = 0.5
max_iterations = 200
input_path = './dataset/images/'
num_batches =  int(np.ceil((len(image_id_list)) / batch_size))
img_size = 299
lr = 2 / 255  # step size
epsilon = 16  # L_inf norm bound
top_k = 10  # the number of class to suppress
T = np.round((6./8) * max_iterations)    # when to introduce the orthogonal loss

# 4. Attacks
# 4.1 CE
setup_seed(42)
pos = np.zeros((3, max_iterations // 50))
confidence = np.zeros((3, max_iterations // 50))
confidence_o = np.zeros((3, max_iterations // 50))
for k in range(0, num_batches):
    print(k)
    batch_size_cur = min(batch_size, len(image_id_list) - k * batch_size)
    X_ori = torch.zeros(batch_size_cur, 3, img_size, img_size).to(device)
    delta = torch.zeros_like(X_ori, requires_grad=True).to(device)
    for i in range(batch_size_cur):
        X_ori[i] = trn(Image.open(input_path + image_id_list[k * batch_size + i] + '.png'))
    labels = torch.tensor(label_tar_list[k * batch_size:k * batch_size + batch_size_cur]).to(device)
    labels_true = torch.tensor(label_ori_list[k * batch_size:k * batch_size + batch_size_cur]).to(device)
    grad_pre = 0
    for t in range(max_iterations):
        X_adv = X_ori + delta
        X_adv_DI, rnd, pad_top, pad_left, c = DI(X_adv)   # DI
        logits = model_2(norm(X_adv_DI))
        loss = nn.CrossEntropyLoss(reduction='sum')(logits, labels)
        loss.backward()
        grad_c = delta.grad.clone()
        grad_c = F.conv2d(grad_c, gaussian_kernel, bias=None, stride=1, padding=(2, 2), groups=3)  # TI
        grad_a = grad_c / torch.mean(torch.abs(grad_c), (1, 2, 3), keepdim=True) + 1 * grad_pre  # MI
        grad_pre = grad_a
        delta.grad.zero_()
        delta.data = delta.data - lr * torch.sign(grad_a)
        delta.data = delta.data.clamp(-epsilon / 255, epsilon / 255)
        delta.data = ((X_ori + delta.data).clamp(0, 1)) - X_ori
        if t % 50 == 49:
            X_adv_norm = norm(X_ori + delta).detach()

            output1 = model_1(X_adv_norm)
            conf1 = F.softmax(output1, dim=-1)
            pos[0, t // 50] = pos[0, t // 50] + sum(torch.argmax(output1, dim=1) == labels).cpu().numpy()
            confidence[0, t // 50] = confidence[0, t // 50] + conf1.gather(1, labels.unsqueeze(1)).sum()
            confidence_o[0, t // 50] = confidence_o[0, t // 50] + conf1.gather(1, labels_true.unsqueeze(1)).sum()

            output3 = model_3(X_adv_norm)
            conf3 = F.softmax(output3, dim=-1)
            pos[1, t // 50] = pos[1, t // 50] + sum(torch.argmax(output3, dim=1) == labels).cpu().numpy()
            confidence[1, t // 50] = confidence[1, t // 50] + conf3.gather(1, labels.unsqueeze(1)).sum()
            confidence_o[1, t // 50] = confidence_o[1, t // 50] + conf3.gather(1, labels_true.unsqueeze(1)).sum()

            output4 = model_4(X_adv_norm)
            conf4 = F.softmax(output4, dim=-1)
            pos[2, t // 50] = pos[2, t // 50] + sum(torch.argmax(output4, dim=1) == labels).cpu().numpy()
            confidence[2, t // 50] = confidence[2, t // 50] + conf4.gather(1, labels.unsqueeze(1)).sum()
            confidence_o[2, t // 50] = confidence_o[2, t // 50] + conf4.gather(1, labels_true.unsqueeze(1)).sum()

torch.cuda.empty_cache()
pos_res50_ce = np.copy(pos)

# 4.2 Po+Trip
setup_seed(42)
pos = np.zeros((3, max_iterations // 50))
confidence = np.zeros((3, max_iterations // 50))
confidence_o = np.zeros((3, max_iterations // 50))
for k in range(0, num_batches):
    print(k)
    batch_size_cur = min(batch_size, len(image_id_list) - k * batch_size)
    X_ori = torch.zeros(batch_size_cur, 3, img_size, img_size).to(device)
    delta = torch.zeros_like(X_ori, requires_grad=True).to(device)
    for i in range(batch_size_cur):
        X_ori[i] = trn(Image.open(input_path + image_id_list[k * batch_size + i] + '.png'))
    labels = torch.tensor(label_tar_list[k * batch_size:k * batch_size + batch_size_cur]).to(device)
    labels_true = torch.tensor(label_ori_list[k * batch_size:k * batch_size + batch_size_cur]).to(device)
    labels_onehot = torch.zeros(batch_size_cur, 1000, device=device)
    labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
    labels_true_onehot = torch.zeros(batch_size_cur, 1000, device=device)
    labels_true_onehot.scatter_(1, labels_true.unsqueeze(1), 1)
    labels_infhot = torch.zeros_like(labels_onehot).scatter_(1, labels.unsqueeze(1), float('inf'))

    grad_pre = 0
    for t in range(max_iterations):
        X_adv = X_ori + delta
        X_adv_DI, rnd, pad_top, pad_left, c = DI(X_adv)  # DI
        logits = model_2(norm(X_adv_DI))

        sum_logits = torch.sum(torch.abs(logits), 1, keepdim=True)
        loss_po = Poincare_dis(logits / sum_logits, torch.clamp((labels_onehot - 0.00001), 0.0, 1.0))
        Cos_dis_dif = Cos_dis(labels_onehot, logits) - Cos_dis(labels_true_onehot, logits) + 0.007
        loss_cos = torch.clamp(Cos_dis_dif, 0.0, 2.1)
        loss = loss_po + 0.01 * loss_cos
        loss.backward()
        grad_c = delta.grad.clone()

        grad_c = F.conv2d(grad_c, gaussian_kernel, bias=None, stride=1, padding=(2, 2), groups=3)  # TI
        grad_a = grad_c + 1 * grad_pre                 # MI
        grad_pre = grad_a
        delta.grad.zero_()
        delta.data = delta.data - lr * torch.sign(grad_a)
        delta.data = delta.data.clamp(-epsilon / 255, epsilon / 255)
        delta.data = ((X_ori + delta.data).clamp(0, 1)) - X_ori
        if t % 50 == 49:
            X_adv_norm = norm(X_ori + delta).detach()

            output1 = model_1(X_adv_norm)
            conf1 = F.softmax(output1, dim=-1)
            pos[0, t // 50] = pos[0, t // 50] + sum(torch.argmax(output1, dim=1) == labels).cpu().numpy()
            confidence[0, t // 50] = confidence[0, t // 50] + conf1.gather(1, labels.unsqueeze(1)).sum()
            confidence_o[0, t // 50] = confidence_o[0, t // 50] + conf1.gather(1, labels_true.unsqueeze(1)).sum()

            output3 = model_3(X_adv_norm)
            conf3 = F.softmax(output3, dim=-1)
            pos[1, t // 50] = pos[1, t // 50] + sum(torch.argmax(output3, dim=1) == labels).cpu().numpy()
            confidence[1, t // 50] = confidence[1, t // 50] + conf3.gather(1, labels.unsqueeze(1)).sum()
            confidence_o[1, t // 50] = confidence_o[1, t // 50] + conf3.gather(1, labels_true.unsqueeze(1)).sum()

            output4 = model_4(X_adv_norm)
            conf4 = F.softmax(output4, dim=-1)
            pos[2, t // 50] = pos[2, t // 50] + sum(torch.argmax(output4, dim=1) == labels).cpu().numpy()
            confidence[2, t // 50] = confidence[2, t // 50] + conf4.gather(1, labels.unsqueeze(1)).sum()
            confidence_o[2, t // 50] = confidence_o[2, t // 50] + conf4.gather(1, labels_true.unsqueeze(1)).sum()

torch.cuda.empty_cache()
pos_res50_trip_po = np.copy(pos)

# 4.3 logits
setup_seed(42)
pos = np.zeros((3, max_iterations // 50))
confidence = np.zeros((3, max_iterations // 50))
confidence_o = np.zeros((3, max_iterations // 50))
for k in range(0, num_batches):
    print(k)
    batch_size_cur = min(batch_size, len(image_id_list) - k * batch_size)
    X_ori = torch.zeros(batch_size_cur, 3, img_size, img_size).to(device)
    delta = torch.zeros_like(X_ori, requires_grad=True).to(device)
    for i in range(batch_size_cur):
        X_ori[i] = trn(Image.open(input_path + image_id_list[k * batch_size + i] + '.png'))
    labels = torch.tensor(label_tar_list[k * batch_size:k * batch_size + batch_size_cur]).to(device)
    labels_true = torch.tensor(label_ori_list[k * batch_size:k * batch_size + batch_size_cur]).to(device)

    grad_pre = 0
    for t in range(max_iterations):
        X_adv = X_ori + delta
        X_adv_DI, rnd, pad_top, pad_left, c = DI(X_adv)
        X_adv_norm_DI = norm(X_adv_DI)
        logits = model_2(X_adv_norm_DI)  # DI
        logit_tar = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
        # logit_ori = logits.gather(1, labels_true.unsqueeze(1)).squeeze(1)
        logit_dists = (-1 * (logit_tar))

        loss = logit_dists.sum()
        loss.backward()
        grad_c = delta.grad.clone()

        grad_c = F.conv2d(grad_c, gaussian_kernel, bias=None, stride=1, padding=(2, 2), groups=3)  # TI
        grad_a = grad_c + 1 * grad_pre  # MI
        grad_pre = grad_a
        delta.grad.zero_()
        delta.data = delta.data - lr * torch.sign(grad_a)
        delta.data = delta.data.clamp(-epsilon / 255, epsilon / 255)
        delta.data = ((X_ori + delta.data).clamp(0, 1)) - X_ori

        if t % 50 == 49:
            X_adv_norm = norm(X_ori + delta).detach()

            output1 = model_1(X_adv_norm)
            conf1 = F.softmax(output1, dim=-1)
            pos[0, t // 50] = pos[0, t // 50] + sum(torch.argmax(output1, dim=1) == labels).cpu().numpy()
            confidence[0, t // 50] = confidence[0, t // 50] + conf1.gather(1, labels.unsqueeze(1)).sum()
            confidence_o[0, t // 50] = confidence_o[0, t // 50] + conf1.gather(1, labels_true.unsqueeze(1)).sum()

            output3 = model_3(X_adv_norm)
            conf3 = F.softmax(output3, dim=-1)
            pos[1, t // 50] = pos[1, t // 50] + sum(torch.argmax(output3, dim=1) == labels).cpu().numpy()
            confidence[1, t // 50] = confidence[1, t // 50] + conf3.gather(1, labels.unsqueeze(1)).sum()
            confidence_o[1, t // 50] = confidence_o[1, t // 50] + conf3.gather(1, labels_true.unsqueeze(1)).sum()

            output4 = model_4(X_adv_norm)
            conf4 = F.softmax(output4, dim=-1)
            pos[2, t // 50] = pos[2, t // 50] + sum(torch.argmax(output4, dim=1) == labels).cpu().numpy()
            confidence[2, t // 50] = confidence[2, t // 50] + conf4.gather(1, labels.unsqueeze(1)).sum()
            confidence_o[2, t // 50] = confidence_o[2, t // 50] + conf4.gather(1, labels_true.unsqueeze(1)).sum()

torch.cuda.empty_cache()
pos_res50_logit = np.copy(pos)

# 4.4 logits + proposed
setup_seed(42)
pos = np.zeros((3, max_iterations // 50))
confidence = np.zeros((3, max_iterations // 50))
confidence_o = np.zeros((3, max_iterations // 50))
for k in range(0, num_batches):
    print(k)
    # A.1: load the images and the corresponding original and target labels
    batch_size_cur = min(batch_size, len(image_id_list) - k * batch_size)
    X_ori = torch.zeros(batch_size_cur, 3, img_size, img_size).to(device)
    delta = torch.zeros_like(X_ori, requires_grad=True).to(device)
    for i in range(batch_size_cur):
        X_ori[i] = trn(Image.open(input_path + image_id_list[k * batch_size + i] + '.png'))
    labels = torch.tensor(label_tar_list[k * batch_size:k * batch_size + batch_size_cur]).to(device)
    labels_true = torch.tensor(label_ori_list[k * batch_size:k * batch_size + batch_size_cur]).to(device)
    ##
    y_high_conf_cur = torch.zeros(batch_size_cur, top_k).long().cuda()

    grad_pre = 0
    for t in range(max_iterations):
        X_adv = X_ori + delta
        # A.2 Input diversity
        X_adv_DI, rnd, pad_top, pad_left, c = DI(X_adv)
        X_adv_norm_DI = norm(X_adv_DI)

        # A.3 Calculate the current logits
        logits = model_2(X_adv_norm_DI)  # DI

        logitstvalue, logitstop = logits.data.topk(top_k + 2, dim=1)
        logit_tar = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
        logit_ori = logits.gather(1, labels_true.unsqueeze(1)).squeeze(1)

        # A.4 Calculate the grad according to the combined-logits (tar & ori) loss
        logit_dists = (-1 * (logit_tar - beta * logit_ori))   # Equation (7) of the paper
        loss = logit_dists.sum()
        loss.backward()
        grad_c = delta.grad.clone()

        # A.5 Important: calculate the high-confidence labels from an intermedia image, not the original image
        if t == T:
            for imindex in range(batch_size_cur):
                k = 0
                for i in range(top_k + 2):
                    # ignore the target label and the original label
                    if (logitstop[imindex, i] == labels_true[imindex]) | (logitstop[imindex, i] == labels[imindex]):
                        continue
                    y_high_conf_cur[imindex, k] = logitstop[imindex, i]
                    k = k + 1
                    if k >= top_k:
                        break

        # A.6 when t>=T, suppressing high-confidence labels
        if t >= T:
            delta.grad.zero_()
            # Assure the resizing and padding parameters are the same in one iteration
            X_adv_DI = DI_pa(X_adv, rnd, pad_top, pad_left, c)
            logits = model_2(norm(X_adv_DI))
            logit_cur = logits.gather(1, y_high_conf_cur)
            # print(y_high_conf_cur)
            logit_dists = (1 * logit_cur).mean(dim=-1)
            loss = logit_dists.sum()   # Equation (8) of the paper
            loss.backward()
            grad2 = delta.grad.data.clone()

            # Calculate the orthogonal component to grad_c, Equation (9) of the paper
            project_grad2 = projAtoB_batch(grad2, grad_c)    

            # combine the orthogonal component of grad2 with grad_c, Equation (10) of the paper
            grad_c = grad_c + beta2 * project_grad2   
        # A.7 Momentum and translation-invariant
        grad_c = F.conv2d(grad_c, gaussian_kernel, bias=None, stride=1, padding=(2, 2), groups=3)  # TI
        grad_a = grad_c + 1 * grad_pre  # MI
        grad_pre = grad_a
        delta.grad.zero_()
        delta.data = delta.data - lr * torch.sign(grad_a)
        delta.data = delta.data.clamp(-epsilon / 255, epsilon / 255)
        delta.data = ((X_ori + delta.data).clamp(0, 1)) - X_ori
        # A.8 Record the results every 50 iterations
        if t % 50 == 49:
            logitstvalue, logitstop = logits.data.topk(10, dim=1)
            X_adv_norm = norm(X_ori + delta).detach()

            output1 = model_1(X_adv_norm)
            conf1 = F.softmax(output1, dim=-1)
            _, out1top = output1.data.topk(10, dim=1)
            pos[0, t // 50] = pos[0, t // 50] + sum(torch.argmax(output1, dim=1) == labels).cpu().numpy()
            confidence[0, t // 50] = confidence[0, t // 50] + conf1.gather(1, labels.unsqueeze(1)).sum()
            confidence_o[0, t // 50] = confidence_o[0, t // 50] + conf1.gather(1, labels_true.unsqueeze(1)).sum()

            output3 = model_3(X_adv_norm)
            conf3 = F.softmax(output3, dim=-1)
            _, out3top = output3.data.topk(10, dim=1)
            pos[1, t // 50] = pos[1, t // 50] + sum(torch.argmax(output3, dim=1) == labels).cpu().numpy()
            confidence[1, t // 50] = confidence[1, t // 50] + conf3.gather(1, labels.unsqueeze(1)).sum()
            confidence_o[1, t // 50] = confidence_o[1, t // 50] + conf3.gather(1, labels_true.unsqueeze(1)).sum()

            output4 = model_4(X_adv_norm)
            conf4 = F.softmax(output4, dim=-1)
            logits4top, out4top = output4.data.topk(10, dim=1)
            pos[2, t // 50] = pos[2, t // 50] + sum(torch.argmax(output4, dim=1) == labels).cpu().numpy()
            confidence[2, t // 50] = confidence[2, t // 50] + conf4.gather(1, labels.unsqueeze(1)).sum()
            confidence_o[2, t // 50] = confidence_o[2, t // 50] + conf4.gather(1, labels_true.unsqueeze(1)).sum()

torch.cuda.empty_cache()
pos_res50_logit_proposed = np.copy(pos)

Done = 1
