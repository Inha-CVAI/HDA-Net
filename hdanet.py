import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import tqdm
import glob
import cv2
import random
from scipy import linalg
from torchvision import transformsa

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
import numpy as np



class CustomDataset(Dataset):  # Change Dataset to CustomDataset
    def __init__(self, dataset_dir, augmentation=None, preprocessing=None, mode=None):
        self.dataset_dir = dataset_dir
        self.samples_paths = glob.glob("{}/*.npy".format(self.dataset_dir))  # Fix the glob pattern

        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.mode = mode

    def __getitem__(self, i):
        sample = np.load(self.samples_paths[i])

        image = (sample[:, :, :3] * 255.0).astype(np.uint8)
        type_mask = sample[:, :, 3]
        nuclear_mask = sample[:, :, 4]

        centroid_prob_mask = self.distance_transform(sample[:, :, 3])

        if np.max(centroid_prob_mask) == 0:
            pass
        else:
            centroid_prob_mask = (centroid_prob_mask / np.max(centroid_prob_mask)) * 1.0

        mask = np.zeros((nuclear_mask.shape[0], nuclear_mask.shape[0], 3))
        mask[:, :, 0] = nuclear_mask
        mask[:, :, 1] = centroid_prob_mask
        mask[:, :, 2] = type_mask

        if self.augmentation:
            try:
                sample = self.augmentation(image=image, mask=mask)
                image, mask = sample["image"], sample["mask"]
            except:
                pass

        # apply preprocessing
        if self.preprocessing:
            image = image / 255.0
            sample = self.preprocessing(image=image)
            image = sample["image"]
        image_blur = cv2.GaussianBlur(image, (5, 5), 0)
        deconv_img = self.deconv_stains(image_blur , self.her_from_rgb)
        Hematoxylin = deconv_img
       
        return image, mask , Hematoxylin

    def __len__(self):
        return len(self.samples_paths)

    def distance_transform(self, inst_mask):
        heatmap = np.zeros_like(inst_mask).astype("uint8")
        for x in np.unique(inst_mask)[1:]:
            temp = inst_mask + 0
            temp = np.where(temp == x, 1, 0).astype("uint8")

            heatmap = heatmap + cv2.distanceTransform(temp, cv2.DIST_L2, 3)

        return heatmap
        # Normalized optical density (OD) matrix M for H and E.
    rgb_from_her = np.array([[0.65, 0.70, 0.29], # H
                            [0.07, 0.99, 0.11], # E
                            [0.00, 0.00, 0.00]])# R
    rgb_from_her[2, :] = np.cross(rgb_from_her[0, :], rgb_from_her[1, :])
    her_from_rgb = linalg.inv(rgb_from_her)

    
    def deconv_stains(self, rgb, conv_matrix):
    
        # change datatype to float64
        rgb = (rgb).astype(np.float64)
        np.maximum(rgb, 1E-6, out=rgb)  # to avoid log artifacts <- 로그 함수는 입력값이 0 에 가까워질수록, 음의 무한대로 수렴, 그래서 0 에 근접하면 아티팩트가 발생할 수 있음
        log_adjust = np.log(1E-6)  # for compensate the sum above
        x = np.log(rgb)
        stains = (x / log_adjust) @ conv_matrix

        # normalizing and shifting the data distribution to proper pixel values range (i.e., [0,255])
        h = 1 - (stains[:,:,0]-np.min(stains[:,:,0]))/(np.max(stains[:,:,0])-np.min(stains[:,:,0]))
        e = 1 - (stains[:,:,1]-np.min(stains[:,:,1]))/(np.max(stains[:,:,1])-np.min(stains[:,:,1]))
        r = 1 - (stains[:,:,2]-np.min(stains[:,:,2]))/(np.max(stains[:,:,2])-np.min(stains[:,:,2]))

        her = cv2.merge((h,e,r)) * 255

        return her.astype(np.uint8) 

def get_validation_augmentation():
    test_transform = [
        
    ]
    return transforms.Compose(test_transform)

def to_tensor(x):
    return torch.from_numpy(x.transpose(2, 0, 1)).float()  # Add the 'return' statement

train_dir = 'Kumar/Train'
test_dir = 'Kumar/Test'

train_dataset = CustomDataset(
    train_dir,
    augmentation=transforms.Compose([
        get_validation_augmentation(),
        lambda x: to_tensor(x)
    ]),
)

test_dataset = CustomDataset(
    test_dir,
    augmentation=transforms.Compose([
        get_validation_augmentation(),
        lambda x: to_tensor(x)
    ])
)
print(f"Training Data Size : {len(train_dataset)}")
print(f"Testing Data Size : {len(test_dataset)}")

from torch.utils.data import DataLoader

 

import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain

class CBR2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super(CBR2d, self).__init__()
        layers = []
        layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                        kernel_size=kernel_size, stride=stride, padding=padding,
                        bias=bias)]
        layers += [nn.BatchNorm2d(num_features=out_channels)]
        layers += [nn.ReLU()]
        self.cbr = nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.cbr(x)
        return out



class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super ( SELayer , self ). __init__ ()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU (),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
""" 3x3->3x3 Residual block """
class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.conv3 = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_c)

        self.se = SELayer(out_c, out_c)
        self.relu  =  nn.ReLU()

    def forward(self, x):
        x1 = self.conv1(x)
        x1  =  self.bn1 (x1)
        x1  =  self.relu (x1)

        x2 = self.conv2(x1)
        x2 = self.bn2(x2)

        x3 = self.conv3(x)
        x3  =  self.bn3(x3)
        x3  =  self.se(x3)

        x4  =  x2 + x3
        x4  =  self.relu(x4)

        return x4


class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(EncoderBlock, self).__init__()
        self.r1 = ResidualBlock(in_c, out_c)
        self.r2 = ResidualBlock(out_c, out_c )
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.r1(inputs)
        x = self.r2(x)
        o = self.pool(x)
        return  o , x

class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c ):
        super(DecoderBlock, self).__init__()

        self.upsample = nn.ConvTranspose2d(in_c, in_c, kernel_size=4, stride=2, padding=1)
        self.r1 = ResidualBlock(in_c+in_c, out_c)
        self.r2 = ResidualBlock(out_c, out_c)

    def forward(self, inputs, skip):
        x = self.upsample(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.r1(x)
        x = self.r2(x)
        return x
    
class CrossAttentionBlock(torch.nn.Module):
    def __init__(self , in_channel):
        super(CrossAttentionBlock, self).__init__()
        self.attention_conv = CBR2d(in_channels = in_channel , out_channels= in_channel , kernel_size=1, stride=1, padding=0 )
    def forward(self, tensor1, tensor2):
        batch , channel , Height , Width = tensor1.shape
        sum_tensor = tensor1 + tensor2
        sum_tensor = F.avg_pool2d(sum_tensor , (Height, Width))
        sum_tensor = self.attention_conv(sum_tensor)
        attention = F.softmax(sum_tensor, dim=1)
        aggregated1 = torch.mul(tensor1, attention)
        aggregated2 = torch.mul(tensor2, attention)

        output = aggregated1 + aggregated2

        return output

            
class FNet(nn.Module):
    def __init__(self):
        super(FNet, self).__init__()
        self.e1 = EncoderBlock(3, 64 ) # ( 128 ,128 , 64)
        self.e2 = EncoderBlock(64, 128) # (64 , 64 , 128)
        self.e3 = EncoderBlock(128, 256 ) # ( 32 , 32 , 256 )
        self.e4 = EncoderBlock(256, 512) # ( 16 , 16 , 512)
        self.e5 = EncoderBlock(512,1024) # (8 , 8 , 1024) 
        self.he1 = EncoderBlock(3, 64 ) # ( 128 ,128 , 64)
        self.he2 = EncoderBlock(64, 128) # (64 , 64 , 128)
        self.he3 = EncoderBlock(128, 256 ) # ( 32 , 32 , 256 )
        self.he4 = EncoderBlock(256, 512) # ( 16 , 16 , 512)
        self.he5 = EncoderBlock(512,1024) # (8 , 8 , 1024) 
        self.d0 = DecoderBlock(1024, 512)
        self.d1 = DecoderBlock(512, 256 )
        self.d2 = DecoderBlock(256, 128 )
        self.d3 = DecoderBlock(128, 64)
        self.d4 = DecoderBlock(64, 5 )
        self.conv = nn.Conv2d(5, 1 , kernel_size= 1 , stride = 1, padding= 0)
        self.conv2 = nn.Conv2d(5, 1 , kernel_size= 1 , stride = 1, padding= 0)
        
        self.CA1 = CrossAttentionBlock(64)
        self.CA2 = CrossAttentionBlock(128)
        self.CA3 = CrossAttentionBlock(256)
        self.CA4 = CrossAttentionBlock(512)
        self.CA5 = CrossAttentionBlock(1024)

    def forward(self, x , y ):
        inputs= x 
        HEcomponents = y
        p1, s1 = self.e1(inputs) # p1 [128, 128 , 64] s1 [256, 256 ,64] 
        hp1, hs1 = self.he1(HEcomponents) # p1 [128, 128 , 64] s1 [256, 256 ,64] 
        en1 = self.CA1(hp1, p1) # en1 [ 128, 128,.64]
        
        
        p2, s2 = self.e2(en1) # p2 [64 , 64 ,128 ] s2 [ 128, 128 , 128]
        hp2, hs2 = self.he2(hp1) # p2 [64 , 64 ,128 ] s2 [ 128, 128 , 128]
        en2 = self.CA2(hp2, p2)
        
        p3, s3  =  self.e3 (en2) # p3 [32, 32 , 256] s3 [64 , 64, 256]
        hp3, hs3  =  self.he3 (hp2) # p3 [32, 32 , 256] s3 [64 , 64, 256]
        en3 = self.CA3(hp3, p3)
        
        
        p4, s4 = self.e4(en3) # p4 [16, 16 ,512] s4 [32 , 32 , 1024]
        hp4, hs4 = self.he4(hp3) # p4 [16, 16 ,512] s4 [32 , 32 , 1024]
        en4 = self.CA4(hp4,p4)
        
        
        p5, s5 = self.e5(en4) # p5 [8 , 8 , 1024] s5 [16 , 16 ,1024]
        hp5, hs5 = self.he5(hp4) # p5 [8 , 8 , 1024] s5 [16 , 16 ,1024]
        en5 = self.CA5(hp5, p5)
        
        
        d0 = self.d0(en5 , s5) # d0 [16, 16 ,512]
        d1 = self.d1(d0, s4) # d1 [32 ,32 ,256]
        d2 = self.d2(d1, s3) # d2 [64, 64 , 128]
        d3 = self.d3(d2, s2) # d3 [128 , 128 , 64]
        d4 = self.d4(d3, s1) # d4 [256, 256 , 5 ]
        output = self.conv(d4) # 
    
        centroid = self.conv2(d4)
        return  output , centroid 

    def get_backbone_params(self):
        # There is no backbone for unet, all the parameters are trained from scratch
        return []

    def get_decoder_params(self):
        return self.parameters()

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()
       
lr = 2e-4
batchsize = 4
num_epoch = 200
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


from torch.utils.data import DataLoader
loader_train = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, drop_last=True ,num_workers=4)
loader_test = DataLoader(test_dataset, batch_size=batchsize, shuffle=False, drop_last=True , num_workers=4)
net = FNet().to(device)

optim = torch.optim.Adam(net.parameters() , lr=  lr)
train_losses = []
val_losses = []
start_epoch = 0


        
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        
        return 1 - dice

for epoch in tqdm(range(start_epoch+1, num_epoch+1)):
    net.train()
    loss_arr = []
    return_mask = []
    for i, batch in enumerate(loader_train):
        images , masks , Hcomp= batch
       
        Hcomp = np.transpose(Hcomp , (0, 3, 1, 2)).to(device)
        Hcomp = Hcomp.float()
        
        nuclear_masks = masks[:,:,:,0].unsqueeze(-1) # [4 , 256 ,256 ,1]
        nuclear_masks = np.transpose(nuclear_masks , (0, 3, 1, 2)).to(device)
        centroid = masks[:, :, : ,1].unsqueeze(-1) #[2 , 256, 256] -> [2,256,256,1]
        centroid = np.transpose(centroid, (0, 3,1,2)).to(device) #[2,1,256,256]
        centroid = centroid.float() # convert to torch.float32 
        images = np.transpose(images, (0, 3, 1, 2)).to(device) # [2,256,256, 3] -> [2, 3, 256 ,256]
        images = images.float() # convert to torch.float32
        nuclear_masks = nuclear_masks.float()
        fn_loss = nn.BCEWithLogitsLoss().to(device)
        optim.zero_grad()
        nuclei_output , centroid_output = net(images , Hcomp)
        
        loss = fn_loss(nuclei_output , nuclear_masks)
        loss2 =fn_loss(centroid_output, centroid)
        loss += loss2
        loss.backward()
        optim.step()
        
      
        loss_arr += [loss.item()]

           
    print("train loss: {} epoch : {}".format(np.mean(loss_arr),epoch))
    train_losses.append(np.mean(loss_arr))


def compute_iou(A,B):
    intersection = np.logical_and(A, B)
    union = np.logical_or(A, B)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score 

def DiceCoefficient(A,B):
    intersection = np.logical_and(A,B)
    Dice_score = 2* np.sum(intersection) / (np.sum(A) + np.sum(B))
    return Dice_score

def precision_score(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    return (intersection + 1e-15) / (y_pred.sum() + 1e-15)

def  recall_score ( y_true , y_pred ):
    intersection = (y_true * y_pred).sum()
    return (intersection + 1e-15) / (y_true.sum() + 1e-15)

def F2_score(y_true, y_pred, beta=2):
    p = precision_score(y_true,y_pred)
    r = recall_score(y_true, y_pred)
    return (1+beta**2.) *(p*r) / float(beta**2*p + r + 1e-15)

# 저장된 모델 가중치 파일 경로
save_dir = ''
import matplotlib.pyplot as plt
# 폴더가 존재하지 않으면 생성
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# 모델 가중치 불러오기


net.eval()
with torch.no_grad():   

        total_loss = 0
        iou_score = 0
        Dice_score = 0
        accuracy = 0
        recall = 0 
        precision = 0
        F1score = 0
        F2score = 0

        
    
        for j,batch in enumerate(loader_test):
             
            with torch.no_grad():
               images , masks ,Hcomp = batch
               Hcomp = np.transpose(Hcomp , (0, 3, 1, 2)).to(device)
               Hcomp = Hcomp.float()
               images = np.transpose(images, (0, 3, 1, 2)).to(device)
               
               nuclear_masks = masks[:,:,:,0].unsqueeze(-1) # [4 , 256 ,256 ,1]
               nuclear_masks = np.transpose(nuclear_masks , (0, 3, 1, 2)).to(device)
               images = images.float() # convert to torch.float32
               nuclear_masks = nuclear_masks.float()
               nuclear_output , centroid_output = net(images, Hcomp)
               
               criterion = nn.BCEWithLogitsLoss()
               

               loss = criterion(nuclear_output, nuclear_masks)

               
               pred2 = nuclear_output
               pred2 = torch.sigmoid(pred2)
               pred2 = (pred2 >= 0.5)
               pred2 = pred2.cpu()
               pred2 = pred2.detach().numpy()
               
               nuclear_masks = nuclear_masks.cpu()
               nuclear_masks = nuclear_masks.detach().numpy()
               
               for k in range(len(images)):
                    input_image = images[k].cpu().numpy() *255  # 이미지 스케일 조정
                    input_path = os.path.join(save_dir, f'input_image_{j}_{k}.png')
                    plt.imsave(input_path, input_image.transpose(1, 2, 0).astype('uint8'))
                    # nuclear_output 이미지 저장
                    output_image = pred2[k, 0, :, :] * 255  # 이미지 스케일 조정
                    output_path = os.path.join(save_dir, f'nuclear_output_{j}_{k}.png')
                    plt.imsave(output_path, output_image.astype('uint8'), cmap='gray')
                    
                    # nuclear_masks 이미지 저장
                    mask_image = nuclear_masks[k, 0, :, :] * 255  # 이미지 스케일 조정
                    mask_path = os.path.join(save_dir, f'nuclear_masks_{j}_{k}.png')
                    plt.imsave(mask_path, mask_image.astype('uint8'), cmap='gray')
               
               total_loss += loss.item()
               
               iou_score += compute_iou(pred2,nuclear_masks)
               Dice_score += DiceCoefficient(pred2,nuclear_masks)
               recall += recall_score(pred2, nuclear_masks)
               precision += precision_score(pred2, nuclear_masks)
               F1score += 2* ((precision_score(pred2,nuclear_masks) * recall_score(pred2,nuclear_masks))  / (precision_score(pred2,nuclear_masks)+recall_score(pred2,nuclear_masks)))
               F2score += F2_score(pred2, nuclear_masks)
         
            
        print('total loss : ' , (total_loss)/(j+1),'IoU : ', iou_score/(j+1) , 'DiceCoefficient : ', Dice_score/(j+1) )
        print('recall : ' , recall/(j+1), 'precision : ' , precision/(j+1) , 'F2 Score : ', F2score/(j+1) ,'F1 score : ', F1score/(j+1)) 
