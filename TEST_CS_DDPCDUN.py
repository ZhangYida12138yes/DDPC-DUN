import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
import numpy as np
import os
import glob
from time import time
import math
from torch.nn import init
import copy
import cv2
from skimage.metrics import structural_similarity as ssim
from argparse import ArgumentParser
import pandas as pd
from utils import *
import csv   

parser = ArgumentParser(description='DDPC-DUN')

parser.add_argument('--layer_num', type=int, default=20, help='stage number of DDPCDUN')
parser.add_argument('--cs_ratio', type=int, default=10, help='from {1, 4, 10, 25, 40, 50}')
parser.add_argument('--noise', type=float, default=0, help='from {1, 4, 10, 25, 40, 50}')
parser.add_argument('--lambda_index', type=int, default=4, help='from {0, 1, 2, 3, 4, 5}')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')
parser.add_argument('--channels', type=int, default=32, help='1 for gray, 3 for color')
parser.add_argument('--matrix_dir', type=str, default='sampling_matrix', help='sampling matrix directory')
parser.add_argument('--model_dir', type=str, default='./model', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='./Dataset', help='training or test data directory')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')
parser.add_argument('--result_dir', type=str, default='./result', help='result directory')
parser.add_argument('--test_name', type=str, default='Set11', help='name of test set')
parser.add_argument('--patch_size', type=int, default=99)
parser.add_argument('--algo_name', type=str, default='DDPCDUN', help='log directory')

args = parser.parse_args()

layer_num = args.layer_num
cs_ratio = args.cs_ratio
gpu_list = args.gpu_list
test_name = args.test_name
channels = args.channels
noise = args.noise
lambda_index = args.lambda_index

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ratio_dict = {1: 10, 4: 43, 5:55, 10: 109, 25: 272, 30: 327, 40: 436, 50: 545}
lambda_list = {0:0.0002, 1:0.0001, 2:0.00004, 3:0.00001, 4:0.000005, 5:0.000001} 

n_input = ratio_dict[cs_ratio]
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

class AdaptiveFuse(nn.Module):
    def __init__(self, in_c=2, out_c=1):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(in_c, 8, 1, padding=0),  
            nn.ReLU(inplace=True),
            nn.Conv2d(8, out_c, 1, padding=0),  
            nn.Sigmoid()  
        )

    def forward(self, x, x2):

        concat = torch.cat([x, x2], dim=1)  
        weight = self.fuse(concat)          
        return x * (1 - weight) + x2 * weight  

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

def gumbel_softmax(x, dim=-1):


    y_soft = x.softmax(dim)
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(x).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft
    return ret


def PhiTPhi_fun(x, PhiW, PhiTW):
    temp = F.conv2d(x, PhiW, padding=0,stride=33, bias=None)
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

    def forward(self, x, x2, z, PhiWeight, PhiTWeight, PhiTb, gate1, gate2, gate4, d_weight, d, D, cs_ratio):
        cs_ratio = torch.full((x.shape[0], 1, 1, 1), cs_ratio, dtype=torch.float32, device=device)
        rho, muz, eta, beta = self.hpn(cs_ratio).chunk(4, dim=1)
        print(f"Shape of x_input: {x.shape}")
        if gate1[:, 1] == 0:
            x = x
        else:
            x = x - rho * PhiTPhi_fun(x, PhiWeight, PhiTWeight)
            x = x + rho * PhiTb

        if gate2[:, 1] == 0:
            x = x
            x2 = x2
        else:
            x2 = x2 + self.pmn[1:-1](self.pmn[:1](x) + self.scale_x2 * x2)  # 这个是alpha自己的更新
            x = x + self.pmn[-1:](x2)
        x_fft = torch.view_as_real(torch.fft.rfft2(x)).unsqueeze(2)
        x2_fft = torch.view_as_real(torch.fft.rfft2(x2)).unsqueeze(1)
        x2 = SolveFFT(x2_fft, D, x_fft, eta, x.shape[-2:])
        x2 = x2 + muz * (x - d(x2))
        if gate4[:, 1] == 0:
            x2 = x2
        else:
            x2 = x2 + self.dcab(x2)
            b = x.shape[0]
            x2 = x2 + self.ptsn(torch.cat([x2, beta[:b].expand_as(x)], dim=1))
        print(f"Shape of x_bbout: {x.shape}")
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
        M = ratio_dict[cs_ratio] 
        

        random_matrix_cpu = torch.randn(M, self.block_size * self.block_size)
        U, S, V = torch.linalg.svd(random_matrix_cpu, full_matrices=False)
        
        w_cpu = (U @ V).view(M, 1, self.block_size, self.block_size)

        self.Phi_weight = nn.Parameter(w_cpu) 

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
        self.fuse_w = nn.Parameter(torch.tensor(0.5))

    def forward(self, Phix_dummy, lamda, cs_ratio): 
        
        self.block_size = 33 
        
        Phix = Phix_dummy 
        
        PhiWeight = self.Phi_weight.contiguous().view(n_input, 1, self.block_size, self.block_size)
    
        Phi_matrix = self.Phi_weight.view(n_input, -1)  
        PhiT_matrix = Phi_matrix.t()                    
        PhiTWeight = PhiT_matrix.view(self.block_size*self.block_size, n_input, 1, 1)  
    
        PhiTb = F.conv2d(Phix, PhiTWeight, padding=0, bias=None)
        PhiTb = torch.nn.PixelShuffle(self.block_size)(PhiTb)
        x = PhiTb
        
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
            
            x, x2 = self.fcs[i](d(x2), x2, z, PhiWeight, PhiTWeight, PhiTb, gate1, gate2, gate4, d_weight, d, D, cs_ratio)
            
            gate1_s.append(gate1[:, 1])
            gate2_s.append(gate2[:, 1])
            gate4_s.append(gate4[:, 1])

        x_final = x
        x_final2 = d(x2)
        gate1_s = torch.cat(gate1_s, 0)
        gate2_s = torch.cat(gate2_s, 0)
        gate4_s = torch.cat(gate4_s, 0)
        x_fused = self.fuse_w * x_final + (1 - self.fuse_w) * x_final2
        
        return x_fused, gate1_s, gate2_s, gate4_s

model = DDPCDUN(layer_num)
model = nn.DataParallel(model)
model = model.to(device)

num_params = 0
for para in model.parameters():
    num_params += para.numel()
print("total para num: %d\n" %num_params)

model_dir = "%s/CS_%s_layer_%d_ratio_%d" % (args.model_dir, args.algo_name, layer_num, cs_ratio)

if cs_ratio==10:
    epoch_num=121
elif cs_ratio==30:
    epoch_num=10
elif cs_ratio == 40:
    epoch_num = 404
elif cs_ratio == 50: 
    epoch_num = 406
elif cs_ratio == 5:
    epoch_num = 150
else:
    epoch_num=403
    

model.load_state_dict(torch.load('%s/net_params_%d.pkl' % (model_dir, epoch_num)))
print('\n')
print("CS Reconstruction Start")

def imread_CS_py(Iorg):
    block_size = 33
    [row, col] = Iorg.shape
    row_pad = block_size-np.mod(row,block_size)
    col_pad = block_size-np.mod(col,block_size)
    Ipad = np.concatenate((Iorg, np.zeros([row, col_pad])), axis=1)
    Ipad = np.concatenate((Ipad, np.zeros([row_pad, col+col_pad])), axis=0)
    [row_new, col_new] = Ipad.shape

    return [Iorg, row, col, Ipad, row_new, col_new]

def psnr(img1, img2):
    img1.astype(np.float32)
    img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

test_dir = os.path.join(args.data_dir, test_name)
if test_name=='Set11':
    filepaths = glob.glob(test_dir + '/*.tif')
else:
    filepaths = glob.glob(test_dir + '/*.png')

result_dir = os.path.join(args.result_dir, test_name)
result_dir = os.path.join(result_dir, str(args.cs_ratio))
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
def test_one_epoch(model, epoch):
    ImgNum = len(filepaths)
    PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
    SSIM_All = np.zeros([1, ImgNum], dtype=np.float32)
    gate1_ALL = np.zeros([1, ImgNum], dtype=np.float32)
    gate2_ALL = np.zeros([1, ImgNum], dtype=np.float32)
    gate4_ALL = np.zeros([1, ImgNum], dtype=np.float32)

    with torch.no_grad():
        for img_no in range(ImgNum):

            imgName = filepaths[img_no]

            Img = cv2.imread(imgName, 1)
            ImgN = imgName.split('/')[-1].split('.')[0]

            Img_yuv = cv2.cvtColor(Img, cv2.COLOR_BGR2YCrCb)
            Img_rec_yuv = Img_yuv.copy()

            Iorg_y = Img_yuv[:,:,0]
            Iorg = Iorg_y.copy()

            [Iorg, row, col, Ipad, row_new, col_new] = imread_CS_py(Iorg_y)
            Img_output = Ipad.reshape(1, 1, Ipad.shape[0], Ipad.shape[1])/255.0

            start = time()

            batch_x = torch.from_numpy(Img_output)
            batch_x = batch_x.type(torch.FloatTensor)
            batch_x = batch_x.to(device)

            Phi_weight = model.module.Phi_weight 
            
            block_size = 33
            PhiWeight = Phi_weight.contiguous().view(n_input, 1, block_size, block_size)
            Phix = F.conv2d(batch_x, PhiWeight, padding=0, stride=block_size, bias=None)
            
            noise_sigma = noise/255.0 * torch.randn_like(Phix)
            Phix = Phix + noise_sigma

            lamblist_encoding = torch.nn.functional.one_hot(torch.tensor([0,1,2,3,4,5]))
            lambda_encoding = lamblist_encoding[lambda_index]
            lambda_value = lambda_list[lambda_index]
            
            x_output, gate1_s, gate2_s, gate4_s = model(Phix, lambda_encoding, cs_ratio)

            end = time()

            Prediction_value = x_output.cpu().data.numpy().squeeze()
            print(f"Shape of Prediction_value: {Prediction_value.shape}")
            row = Iorg.shape[0]
            col = Iorg.shape[1]

            gates1 = gate1_s.cpu().data.numpy().squeeze()
            gates2 = gate2_s.cpu().data.numpy().squeeze()
            gates4 = gate4_s.cpu().data.numpy().squeeze()
        
            X_rec = np.clip(Prediction_value[:row, :col], 0, 1)  
            print(f"Shape of X_rec: {X_rec.shape}")
        
            rec_PSNR = psnr(X_rec*255, Iorg.astype(np.float64))
            rec_SSIM = ssim(X_rec*255, Iorg.astype(np.float64), data_range=255)

            print("[%02d/%02d] Run time for %s is %.4f, sum_gate1 is %d, sum_gate2 is %d, sum_gate4 is %d, PSNR is %.2f, SSIM is %.4f" % (img_no, ImgNum, imgName, (end - start), np.sum(gates1), np.sum(gates2), np.sum(gates4), rec_PSNR, rec_SSIM))

            Img_rec_yuv[:,:,0] = X_rec*255

            im_rec_rgb = cv2.cvtColor(Img_rec_yuv, cv2.COLOR_YCrCb2BGR)
            im_rec_rgb = np.clip(im_rec_rgb, 0, 255).astype(np.uint8)

            cv2.imwrite("%s/%s_%s_lambda_%.5f_ratio_%d_PSNR_%.2f_SSIM_%.4f.png" % (result_dir, ImgN, args.algo_name, lambda_value, cs_ratio, rec_PSNR, rec_SSIM), im_rec_rgb)
            del x_output

            PSNR_All[0, img_no] = rec_PSNR
            SSIM_All[0, img_no] = rec_SSIM
            gate1_ALL[0, img_no] = np.sum(gates1)
            gate2_ALL[0, img_no] = np.sum(gates2)
            gate4_ALL[0, img_no] = np.sum(gates4)

        print('\n')
        output_data = "CS ratio is %d, lambda is %.5f, Avg PSNR/SSIM for %s is %.2f/%.4f, Avg gate1/gate2/gate4 is %d/%d/%d, Epoch number of model is %d \n" % (cs_ratio, lambda_list[lambda_index], args.test_name, np.mean(PSNR_All), np.mean(SSIM_All), np.mean(gate1_ALL), np.mean(gate2_ALL), np.mean(gate4_ALL), epoch_num)
        print(output_data)
    
        return np.mean(PSNR_All), np.mean(SSIM_All),np.mean(gate1_ALL), np.mean(gate2_ALL), np.mean(gate4_ALL)

start_epoch = 1
end_epoch   = 150
result_csv  = os.path.join(result_dir,
                           'all_epoch_PSNR_SSIM_%s_ratio_%d_lambda_%.5f.csv'
                           % (args.algo_name, cs_ratio, lambda_list[lambda_index]))

with open(result_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'avg_PSNR', 'avg_SSIM', 'avg_gate1', 'avg_gate2', 'avg_gate4'])

    for epoch in range(start_epoch, end_epoch + 1):

        if (epoch % decay_step_epoch) == 0:
        
            if epoch > start_epoch: 
            
                print(f"\n--- Epoch {epoch}: 正在衰减 Lambda 权重 (用于测试) ---")
            
                for key in lambda_list:
                    lambda_list[key] *= lambda_decay_gamma
                
                lambda_fft_weight *= lambda_decay_gamma
            
                print(f"新的 Lambda List (Gate): {lambda_list}")
                print(f"新的 Lambda_fft 权重: {lambda_fft_weight:.6f}")
        
        weight_file = '%s/net_params_%d.pkl' % (model_dir, epoch)
        if not os.path.isfile(weight_file):
            print('⚠️  skip  %s' % weight_file)
            continue

        model.load_state_dict(torch.load(weight_file, map_location=device))
        model.eval()

        psnr_avg, ssim_avg, g1, g2, g4 = test_one_epoch(model, epoch)

        writer.writerow([epoch,
                         f'{psnr_avg:.2f}',
                         f'{ssim_avg:.4f}',
                         int(g1), int(g2), int(g4)])
        f.flush()
        print('✅ epoch %3d  |  PSNR %.2f  SSIM %.4f' % (epoch, psnr_avg, ssim_avg))

print('\n------ 全部测试完成，结果已写入 ------')
print(result_csv)
print("CS Reconstruction End")
