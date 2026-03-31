# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import scipy.io as sio
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import platform
from argparse import ArgumentParser
import random
import csdata_fast
import copy
from utils import *

parser = ArgumentParser(description='DDPC-DUN')

parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=150, help='epoch number of end training')
parser.add_argument('--finetune', type=int, default=10, help='epoch number of finetuning')
parser.add_argument('--layer_num', type=int, default=20, help='stage number of DDPCDUN')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--cs_ratio', type=int, default=30, help='from {10, 25, 30, 40, 50}')

parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')
parser.add_argument('--patch_size', type=int, default=33)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--rgb_range', type=int, default=1, help='value range 1 or 255')
parser.add_argument('--n_channels', type=int, default=1, help='1 for gray, 3 for color')
parser.add_argument('--channels', type=int, default=32, help='number of feature map')

parser.add_argument('--matrix_dir', type=str, default='sampling_matrix', help='sampling matrix directory')
parser.add_argument('--model_dir', type=str, default='./model', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='./Dataset', help='training data directory')
parser.add_argument('--train_name', type=str, default='train400', help='name of test set')
parser.add_argument('--ext', type=str, default='.png', help='training data directory')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')
parser.add_argument('--algo_name', type=str, default='DDPCDUN', help='log directory')
parser.add_argument('--data_copy', type=int, default=200)

args = parser.parse_args()

start_epoch = args.start_epoch
end_epoch = args.end_epoch
learning_rate = args.learning_rate
layer_num = args.layer_num
cs_ratio = args.cs_ratio
gpu_list = args.gpu_list
channels = args.channels
finetune = args.finetune
batch_size = args.batch_size

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ratio_dict = {10: 0, 25: 1, 30: 2, 40: 3, 50: 4, 5: 5}
n_input_dict = {1: 10, 4: 43, 5: 55, 10: 109, 20: 218, 30: 327, 40: 436, 50: 545}
lambda_list = {0: 0.0002, 1: 0.0001, 2: 0.00004, 3: 0.00001, 4: 0.000005, 5: 0.000001}

n_input = n_input_dict[cs_ratio]
n_output = 1089


lambda_decay_gamma = 0.95

decay_step_epoch = 1

lambda_fft_weight = 0.01


class SimpleLayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight * x + self.bias
        return x


class FusionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FusionLayer, self).__init__()
        self.conv_x = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x, x2):
        self.conv_x = self.conv_x.to(x.device)
        self.conv = self.conv.to(x.device)
        x = self.conv_x(x)  
        combined = x + x2  
        output = self.conv(combined)  
        x = output
        return x


def PhiTPhi_fun(x, PhiW, PhiTW):
    temp = F.conv2d(x, PhiW, padding=0, stride=33, bias=None)
    temp = F.conv2d(temp, PhiTW, padding=0, bias=None)
    return torch.nn.PixelShuffle(33)(temp)


class Lambda_Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Lambda_Conv, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False)
        self.fc2 = nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False)
        self.conv = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=True)

    def forward(self, x, x2, lam):
        lam = lam.unsqueeze(-1).unsqueeze(-1)
        s = self.fc1(lam)
        s = F.softplus(s)
        b = self.fc2(lam)
        x = FusionLayer(x.shape[1], x2.shape[1])(x, x2)
        x = s * self.conv(x) + b
        return x


class Attention_SEblock(nn.Module):
    def __init__(self, channels, reduction):
        super(Attention_SEblock, self).__init__()
        self.conv = Lambda_Conv(6, channels)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, 2)
        self.fc3 = nn.Linear(channels // reduction, 2)
        self.fc5 = nn.Linear(channels // reduction, 2)
        self.fc2.bias.data[0] = 0.1
        self.fc2.bias.data[1] = 2
        self.fc3.bias.data[0] = 0.1
        self.fc3.bias.data[1] = 2
        self.fc5.bias.data[0] = 0.1
        self.fc5.bias.data[1] = 2
        self.channels = channels

    def forward(self, x, x2, lam):
        x = self.conv(x, x2, lam)
        x = self.avg_pool(x).view(-1, self.channels)
        x = self.fc1(x)
        x = self.relu(x)
        x_2 = self.fc2(x)
        x_2 = F.gumbel_softmax(x_2, tau=1, hard=True)
        x_3 = self.fc3(x)
        x_3 = F.gumbel_softmax(x_3, tau=1, hard=True)
        x_5 = self.fc5(x)
        x_5 = F.gumbel_softmax(x_5, tau=1, hard=True)
        return x_2, x_3, x_5


class ResidualBlock(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(nf, nf, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(nf, nf, 3, padding=2, dilation=2),
        )

    def forward(self, x):
        return x + self.body(x)


class BasicBlock(torch.nn.Module):
    def __init__(self, nf=32, nb=2, reduction=8):
        super(BasicBlock, self).__init__()
        rb_num = 2
        self.conv_D = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))

        ##改动
        self.hpn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=1), 
            nn.Sigmoid(),  
            nn.Conv2d(32, 4, kernel_size=1), 
            nn.Softplus()  
        )
        self.pmn = nn.Sequential(
            nn.Conv2d(1, nf, 3, padding=1),
            *[ResidualBlock(nf) for _ in range(nb)],
            nn.Conv2d(nf, 1, 3, padding=1),
        )
        self.ptsn = nn.Sequential(
            nn.Conv2d(1 + nf, nf, 3, padding=1),
            *[ResidualBlock(nf) for _ in range(nb)],
        )
        self.scale_x2 = nn.Parameter(torch.tensor([0.0]))

        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(nf, nf // reduction, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(nf // reduction, nf, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        self.ln = SimpleLayerNorm(nf)  

        self.dcab = nn.Sequential(
            nn.Conv2d(nf, nf, kernel_size=3, padding=1, groups=nf),  
            self.ln,  
            nn.Conv2d(nf, 4 * nf, kernel_size=1, padding=0), 
            nn.GELU(),
            nn.Conv2d(4 * nf, nf, kernel_size=1, padding=0), 
            self.ca,  
        )

    def forward(self, x, x2, z, PhiWeight, PhiTWeight, PhiTb, gate1, gate2, gate4, d_weight, d, D,
                cs_ratio):
        cs_ratio = torch.full((32, 1, 1, 1), cs_ratio, dtype=torch.float32, device=device)
        rho, muz, eta, beta = self.hpn(cs_ratio).chunk(4, dim=1)

        x_gate1 = gate1[:, 1].view(-1, 1, 1, 1)
        x_gate2 = gate2[:, 1].view(-1, 1, 1, 1)
        x_gate4 = gate4[:, 1].view(-1, 1, 1, 1)


        x = x - rho * x_gate1 * PhiTPhi_fun(x, PhiWeight, PhiTWeight)
        x = x + rho * x_gate1 * PhiTb

        x2 = x2 + x_gate2 * self.pmn[1:-1](self.pmn[:1](x) + self.scale_x2 * x2) 
        x = x + x_gate2 * self.pmn[-1:](x2)

        x_fft = torch.view_as_real(torch.fft.rfft2(x)).unsqueeze(2)
        x2_fft = torch.view_as_real(torch.fft.rfft2(x2)).unsqueeze(1)
        x2 = SolveFFT(x2_fft, D, x_fft, eta, x.shape[-2:])  
        x2 = x2 + muz * (x - d(x2))
        x2 = x2 + x_gate4 * self.dcab(x2)
        x2 = x2 + x_gate4 * self.ptsn(torch.cat([x2, beta.expand_as(x)], dim=1))
        return x, x2

class DDPCDUN(torch.nn.Module):
    def __init__(self, LayerNo, k=5, nf=32):
        super(DDPCDUN, self).__init__()
        onelayer = []
        gates = []
        self.LayerNo = LayerNo
        n_feat = channels - 1

        self.d_weight = nn.Parameter(torch.zeros(1, nf, k, k))
        self.d = lambda w: F.conv2d(w, self.d_weight.to(w.device), padding=k // 2)

        self.block_size = 33
        M = n_input_dict[cs_ratio]

        random_matrix_cpu = torch.randn(M, self.block_size * self.block_size)
        U, S, V = torch.linalg.svd(random_matrix_cpu, full_matrices=False)

        w_cpu = (U @ V).view(M, 1, self.block_size, self.block_size)
        self.Phi_weight = nn.Parameter(w_cpu.to(device))  

        for i in range(LayerNo):
            onelayer.append(BasicBlock())
        self.fcs = nn.ModuleList(onelayer)
        for i in range(LayerNo):
            gates.append(Attention_SEblock(channels, 4))
        self.gates = nn.ModuleList(gates)

        self.fe = nn.Conv2d(1, n_feat, 3, padding=1, bias=True)

        self.d_weight = nn.Parameter(torch.zeros(1, nf, k, k))

        nf_branch = 8
        in_channels = 2

        self.msi_relu = nn.ReLU(True)

        self.msi_conv_pre = nn.Conv2d(in_channels, nf_branch, 1, padding=0)

        self.msi_branch1 = nn.Conv2d(nf_branch, nf_branch, 3, padding=1)  
        self.msi_branch2 = nn.Conv2d(nf_branch, nf_branch, 5, padding=2)  
        self.msi_branch3 = nn.Conv2d(nf_branch, nf_branch, 7, padding=3)  
        self.msi_branch4 = nn.Conv2d(nf_branch, nf_branch, 3, padding=2, dilation=2) 

    def forward(self, x, lamda, cs_ratio):

        Phix = F.conv2d(x, self.Phi_weight, stride=self.block_size, padding=0)

        PhiWeight = self.Phi_weight.contiguous().view(n_input, 1, self.block_size, self.block_size)

        Phi_matrix = self.Phi_weight.view(n_input, -1)  # 4D → 2D: [n_input, 1089]
        PhiT_matrix = Phi_matrix.t()  
        PhiTWeight = PhiT_matrix.view(self.block_size * self.block_size, n_input, 1,
                                      1)  # 2D → 4D: [1089, n_input, 1, 1]

        PhiTb = F.conv2d(Phix, PhiTWeight, padding=0, bias=None)
        PhiTb = torch.nn.PixelShuffle(self.block_size)(PhiTb)

        x = PhiTb
        self.fuse_w = nn.Parameter(torch.tensor(0.5, device=device))  
        b = x.shape[0]
        cs_ratio1 = torch.full((b,), cs_ratio, dtype=torch.float32, device=device)
        cs_ratio1 = cs_ratio1.reshape(b, 1, 1, 1)
        z = self.fe(x)
        x2 = torch.cat([x, cs_ratio1.expand_as(x)], dim=1)


        x2 = self.msi_conv_pre(x2)  
        x2 = self.msi_relu(x2)


        b1 = self.msi_branch1(x2)
        b2 = self.msi_branch2(x2)
        b3 = self.msi_branch3(x2)
        b4 = self.msi_branch4(x2)

        x2 = torch.cat([b1, b2, b3, b4], dim=1)

        lamda = lamda.repeat(x.shape[0], 1).type(torch.FloatTensor).to(device)
        gate1_s = []
        gate2_s = []
        gate4_s = []
        d_weight, d = self.d_weight, self.d
        D = p2o(d_weight.unsqueeze(0), x.shape[-2:])

        for i in range(self.LayerNo):
            if i == 0:
                gate1, gate2, gate4 = self.gates[i](x, x2, lamda)
            else:
                gate1, gate2, gate4 = self.gates[i](x, x2, lamda)
            x, x2 = self.fcs[i](d(x2), x2, z, PhiWeight, PhiTWeight, PhiTb, gate1, gate2, gate4, d_weight, d, D,
                                cs_ratio)

            gate1_s.append(gate1[:, 1])
            gate2_s.append(gate2[:, 1])
            gate4_s.append(gate4[:, 1])

        x_final = x
        x_final2 = d(x2)
        x_fused = self.fuse_w * x_final + (1 - self.fuse_w) * x_final2

        return x_fused, gate1_s, gate2_s, gate4_s


model = DDPCDUN(layer_num)
model = nn.DataParallel(model)
model = model.to(device)

print_flag = 1  # print parameter number

if print_flag:
    num_count = 0
    num_params = 0
    for para in model.parameters():
        num_count += 1
        num_params += para.numel()
        print('Layer %d' % num_count)
        print(para.size())
    print("total para num: %d" % num_params)


class RandomDataset(Dataset):
    def __init__(self, data, length):
        self.data = data
        self.len = length

    def __getitem__(self, index):
        return torch.Tensor(self.data[index, :]).float()

    def __len__(self):
        return self.len


training_data = csdata_fast.SlowDataset(args)

if (platform.system() == "Windows"):
    rand_loader = DataLoader(dataset=training_data, batch_size=batch_size, num_workers=0,
                             shuffle=True)
else:
    rand_loader = DataLoader(dataset=training_data, batch_size=batch_size, num_workers=8,
                             shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model_dir = "%s/CS_%s_layer_%d_ratio_%d" % (args.model_dir, args.algo_name, layer_num, cs_ratio)
log_file_name = "%s/Log_CS_%s_layer_%d_ratio_%d.txt" % (model_dir, args.algo_name, layer_num, cs_ratio)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if start_epoch > 0:
    pre_model_dir = model_dir
    model.load_state_dict(torch.load('%s/net_params_%d.pkl' % (pre_model_dir, start_epoch)))

media_epoch = end_epoch
if finetune > 0:
    end_epoch = end_epoch + finetune
    patch_size1 = 99

# Training loop
for epoch_i in range(start_epoch + 1, end_epoch + 1):

    if (epoch_i % decay_step_epoch) == 0:

        if epoch_i > start_epoch:

            print(f"\n--- Epoch {epoch_i}: 开始阶梯衰减 Lambda 权重 ---")

            for key in lambda_list:
                lambda_list[key] *= lambda_decay_gamma

            lambda_fft_weight *= lambda_decay_gamma

            print(f"新的 Lambda List (Gate): {lambda_list}")
            print(f"新的 Lambda_fft 权重: {lambda_fft_weight:.6f}")

    if epoch_i > media_epoch:
        args.patch_size = patch_size1

    for data in rand_loader:

        batch_x = data
        batch_x = batch_x.to(device)
        batch_x = batch_x.view(-1, 1, args.patch_size, args.patch_size)

        lambda_index = random.randint(0, 5)
        lamblist_encoding = torch.nn.functional.one_hot(torch.tensor([0, 1, 2, 3, 4, 5]))
        lambda_encoding = lamblist_encoding[lambda_index]
        lambda_value = lambda_list[lambda_index]

        x_output, gate1_s, gate2_s, gate4_s = model(batch_x, lambda_encoding, cs_ratio)

        # Compute and print loss
        loss_gate1 = 0
        loss_gate2 = 0
        loss_gate4 = 0
        for i in range(layer_num):
            loss_gate1 = loss_gate1 + gate1_s[i]
            loss_gate2 = loss_gate2 + gate2_s[i]
            loss_gate4 = loss_gate4 + gate4_s[i]
        loss_gate1 = torch.mean((loss_gate1 + 1e-6) / layer_num)
        loss_gate2 = torch.mean((loss_gate2 + 1e-6) / layer_num)
        loss_gate4 = torch.mean((loss_gate4 + 1e-6) / layer_num)

        loss_rec = nn.L1Loss()(x_output, batch_x)
        gt_fft = torch.fft.rfft2(batch_x)
        pred_fft = torch.fft.rfft2(x_output)
        loss_fft = F.l1_loss(pred_fft, gt_fft)

        loss_all = lambda_value * (loss_gate1 + loss_gate2 + loss_gate4) + loss_rec + lambda_fft_weight * loss_fft

        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()

        output_data = "[%02d/%02d] Loss_gate1: %.4f, Loss_gate2: %.4f, Loss_gate4: %.4f, Loss_rec: %.4f, Loss_fft: %.4f\n" % (
            epoch_i, end_epoch, loss_gate1.item(), loss_gate2.item(), loss_gate4.item(),
            loss_rec.item(), loss_fft.item())
        print(output_data)

    output_file = open(log_file_name, 'a')
    output_file.write(output_data)
    output_file.close()

    torch.save(model.state_dict(), "%s/net_params_%d.pkl" % (model_dir, epoch_i))  