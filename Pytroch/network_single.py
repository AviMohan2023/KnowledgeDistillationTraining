import torch
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from scipy import signal
import scipy
from torch import nn
import time


def dft_conv(imgR, imgIm, kernelR, kernelIm):
    # Fast complex multiplication
    #print(kernelR.shape, imgR.shape)
    ac = torch.mul(kernelR, imgR)
    bd = torch.mul(kernelIm, imgIm)

    ab_cd = torch.mul(torch.add(kernelR, kernelIm), torch.add(imgR, imgIm))
    # print(ab_cd.sum(1)[0,0,:,:])
    imgsR = ac - bd
    imgsIm = ab_cd - ac - bd

    # Sum over in channels
    imgsR = imgsR.sum(1)
    imgsIm = imgsIm.sum(1)

    return imgsR, imgsIm


class FFT_Conv_Layer(nn.Module):

    def __init__(self, imgSize, inCs, outCs1, outCs2, outCs3, imagDim, filtSize, cuda=False):
        super(FFT_Conv_Layer, self).__init__()
        self.filts1 = np.random.normal(0, 0.01, (inCs, outCs1, imagDim, filtSize, filtSize))
        self.filts2 = np.random.normal(0, 0.01, (outCs1, outCs2, imagDim, filtSize, filtSize))
        self.filts3 = np.random.normal(0, 0.01, (outCs2, outCs3, imagDim, filtSize, filtSize))
        self.imgSize = imgSize
        self.filtSize = np.size(self.filts1, 3)
        self.pad = nn.ConstantPad2d(((imgSize-4)//2), 0)

        if cuda:
            self.filts1 = torch.from_numpy(self.filts1).type(torch.float32).cuda()
            self.filts1 = Parameter(self.filts1)

            self.filts2 = torch.from_numpy(self.filts2).type(torch.float32).cuda()
            self.filts2 = Parameter(self.filts2)

            self.filts3 = torch.from_numpy(self.filts3).type(torch.float32).cuda()
            self.filts3 = Parameter(self.filts3)

    def forward(self, imgs):
        # Pad and transform the image
        # Pad arg = (last dim pad left side, last dim pad right side, 2nd last dim left side, etc..)

        imgs = imgs.unsqueeze(2)
        imgs = imgs.unsqueeze(5)

        #imgs = F.pad(imgs, (0, 0, 0, self.filtSize - 1, 0, self.filtSize - 1))
        imgs = imgs.squeeze(5)

        imgs = torch.rfft(imgs, 2, onesided=False)
        # print(imgs.shape)

        # Extract the real and imaginary parts
        imgsR = imgs[:, :, :, :, :, 0]
        imgsIm = imgs[:, :, :, :, :, 1]

        # Pad and transform the filters
        filts1 = self.pad(self.filts1)
        filts1shape = filts1.shape
        filts1 = filts1.view(filts1shape[0],filts1shape[1],filts1shape[3],filts1shape[4],filts1shape[2])
        filts1 = filts1.unsqueeze(0)

        filts2 = self.pad(self.filts2)
        filts2shape = filts2.shape
        filts2 = filts2.view(filts2shape[0], filts2shape[1], filts2shape[3], filts2shape[4], filts2shape[2])
        filts2 = filts2.unsqueeze(0)

        filts3 = self.pad(self.filts3)
        filts3shape = filts3.shape
        filts3 = filts3.view(filts3shape[0], filts3shape[1], filts3shape[3], filts3shape[4], filts3shape[2])
        filts3 = filts3.unsqueeze(0)

        filts1 = torch.fft(filts1, 2)
        filts2 = torch.fft(filts2, 2)
        filts3 = torch.fft(filts3, 2)

        # Extract the real and imaginary parts
        filt1R = filts1[:, :, :, :, :, 0]
        filt1Im = filts1[:, :, :, :, :, 1]

        filt2R = filts2[:, :, :, :, :, 0]
        filt2Im = filts2[:, :, :, :, :, 1]

        filt3R = filts3[:, :, :, :, :, 0]
        filt3Im = filts3[:, :, :, :, :, 1]

        # Do element wise complex multiplication
        imgsR, imgsIm = dft_conv(imgsR, imgsIm, filt1R, filt1Im)
        imgsR = imgsR.unsqueeze(2)
        imgsIm = imgsIm.unsqueeze(2)
        imgsR, imgsIm = dft_conv(imgsR, imgsIm, filt2R, filt2Im)
        imgsR = imgsR.unsqueeze(2)
        imgsIm = imgsIm.unsqueeze(2)
        imgsR, imgsIm = dft_conv(imgsR, imgsIm, filt3R, filt3Im)

        # Add dim to concat over
        imgsR = imgsR.unsqueeze(4)
        imgsIm = imgsIm.unsqueeze(4)

        # Concat the real and imaginary again then IFFT
        imgs = torch.cat((imgsR, imgsIm), -1)
        # print("1",imgs.shape)
        imgs = torch.ifft(imgs, 2)
        # print("2",imgs.shape)

        # Filter and imgs were real so imag should be ~0
        imgs = imgs[:, :, 1:-1, 1:-1, 0]
        # print("3",imgs.shape)
        return imgs


class StudentNetwork_noRelu(nn.Module):
    def __init__(self,in_channels):
        super(StudentNetwork_noRelu, self).__init__()
        self.conv1 = FFT_Conv_Layer(imgSize=28, inCs=in_channels, outCs1=32, outCs2=128, outCs3=256, imagDim=2, filtSize=4, cuda=True)

        self.conv2_bn = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(9216, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.dropout_input = 0.5
        self.dropout_hidden = 0.5
        self.is_training = True
        self.avepool = nn.AdaptiveAvgPool2d((6, 6))
        self.m = nn.LogSoftmax(dim=1)


    def forward(self, x):
        forw = self.conv1(x)


        forw = self.conv2_bn(forw)
        #forw = self.maxpool(forw)
        forw = self.avepool(forw)
        forw = forw.view(-1, 9216)
        forw = F.dropout(forw, p=self.dropout_input, training=self.is_training)
        forw = F.dropout(self.fc1(forw), p=self.dropout_hidden, training=self.is_training)
        forw = F.relu(forw)
        forw = self.fc2(forw)
        forw = F.relu(forw)
        forw = self.fc3(forw)
        return self.m(forw)


class Teacher_Network(nn.Module):
    def __init__(self, in_channels):
        super(Teacher_Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)

        self.conv2_bn = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(9216, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

        self.dropout_input = 0.5
        self.dropout_hidden = 0.5
        self.is_training = True
        self.avepool = nn.AdaptiveAvgPool2d((6, 6))
        self.m = nn.LogSoftmax(dim=1)

    def forward(self, x):
        forw = nn.functional.relu(self.conv1(x))
        forw = nn.functional.relu(self.conv2(forw))
        forw = nn.functional.relu(self.conv3(forw))

        forw = self.conv2_bn(forw)
        forw = self.avepool(forw)
        forw = forw.view(-1, 9216)
        forw = F.dropout(forw, p=self.dropout_input, training=self.is_training)
        forw = F.dropout(self.fc1(forw), p=self.dropout_hidden, training=self.is_training)
        forw = F.relu(forw)
        forw = self.fc2(forw)
        forw = F.relu(forw)
        forw = self.fc3(forw)
        return self.m(forw)


class Student_Network_one_conv(nn.Module):
    def __init__(self, in_channels):
        super(Student_Network_one_conv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=6,bias=False)

        self.conv2_bn = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(9216, 10)

        self.dropout_input = 0.5
        self.dropout_hidden = 0.5
        self.is_training = True
        self.avepool = nn.AdaptiveAvgPool2d((6, 6))
        self.m = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # print(x.shape)
        forw = nn.functional.relu(self.conv1(x))
        # print(forw.shape)

        forw = self.conv2_bn(forw)
        forw = self.avepool(forw)
        forw = forw.view(-1, 9216)
        # forw = F.dropout(forw, p=self.dropout_input, training=self.is_training)
        # forw = F.dropout(self.fc1(forw), p=self.dropout_hidden, training=self.is_training)
        # forw = F.relu(forw)
        # forw = self.fc2(forw)
        # forw = F.relu(forw)
        forw = self.fc1(forw)
        return self.m(forw)

if __name__ == '__main__':
    # test teacher
    test_net = Student_Network_one_conv(in_channels=1)
    x = torch.zeros([16, 1, 28, 28])
    out = test_net(x)
    print(out.shape)
    #
    # # test student
    # torch.cuda.set_device(1)
    # test_net = Student_Network_one_conv().cuda()
    # x = torch.zeros([8, 3, 28, 28]).cuda()
    # test_net(x)
