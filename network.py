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

    def __init__(self, imgSize, inCs, outCs, imagDim, filtSize, cuda=False):
        super(FFT_Conv_Layer, self).__init__()
        self.filts = np.random.normal(0, 0.01, (1, inCs, outCs, filtSize, filtSize, imagDim))
        self.imgSize = imgSize
        self.filtSize = np.size(self.filts, 4)

        if cuda:
            self.filts = torch.from_numpy(self.filts).type(torch.float32).cuda()
            self.filts = Parameter(self.filts)

    def forward(self, imgs):
        # Pad and transform the image
        # Pad arg = (last dim pad left side, last dim pad right side, 2nd last dim left side, etc..)
        # imgs = torch.randn(batchSize,inCs,1,imgSize, imgSize,imagDim).cuda()
        imgs = imgs.unsqueeze(2)
        imgs = imgs.unsqueeze(5)

        imgs = F.pad(imgs, (0, 0, 0, self.filtSize - 1, 0, self.filtSize - 1))
        imgs = imgs.squeeze(5)

        imgs = torch.rfft(imgs, 2, onesided=False)
        # print(imgs.shape)

        # Extract the real and imaginary parts
        imgsR = imgs[:, :, :, :, :, 0]
        imgsIm = imgs[:, :, :, :, :, 1]

        # Pad and transform the filters
        filts = F.pad(self.filts, (0, 0, 0, self.imgSize - 1, 0, self.imgSize - 1))

        filts = torch.fft(filts, 2)

        # Extract the real and imaginary parts
        filtR = filts[:, :, :, :, :, 0]
        filtIm = filts[:, :, :, :, :, 1]

        # Do element wise complex multiplication
        imgsR, imgsIm = dft_conv(imgsR, imgsIm, filtR, filtIm)

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

class spectral_pool_layer(nn.Module):
    def __init__(self, filter_size=3, freq_dropout_lower_bound=None, freq_dropout_upper_bound=None, train_phase=False):
        super(spectral_pool_layer, self).__init__()
        # assert only 1 dimension passed for filter size
        assert isinstance(filter_size, int)
        # input_shape = x.shape
        # assert len(input_shape) == 4
        # _, _, H, W = input_shape
        # assert H == W

        self.filter_size = filter_size
        self.freq_dropout_lower_bound = freq_dropout_lower_bound
        self.freq_dropout_upper_bound = freq_dropout_upper_bound
        self.activation = F
        self.train_phase = train_phase

    def forward(self, x):
        # Compute the Fourier transform of the image
        im_fft = torch.rfft(x, 2, onesided=False)

        # Truncate the spectrum
        im_transformed = self._common_spectral_pool(im_fft, self.filter_size)

        if (self.freq_dropout_lower_bound is not None and self.freq_dropout_upper_bound is not None):
            def true_fn():
                tf_random_cutoff = tf.random_uniform(
                    [],
                    freq_dropout_lower_bound,
                    freq_dropout_upper_bound
                )
                dropout_mask = _frequency_dropout_mask(
                    filter_size,
                    tf_random_cutoff
                )
                return im_transformed * dropout_mask

            # In the testing phase, return the truncated frequency
            # matrix unchanged.
            def false_fn():
                return im_transformed

            im_downsampled = tf.cond(
                self.train_phase,
                true_fn=true_fn,
                false_fn=false_fn
            )
            im_out = torch.irfft(im_downsampled, 2, onesided=False)

        else:
            im_out = torch.irfft(im_transformed, 2, onesided=False)

        if self.activation is not None:
            cell_out = self.activation.relu(im_out)
        else:
            cell_out = im_out
        return cell_out

    def _common_spectral_pool(self, images, filter_size):
        assert len(images.shape) == 5
        assert filter_size >= 3

        if filter_size % 2 == 1:
            n = int((filter_size - 1) / 2)
            top_left = images[:, :, :n + 1, :n + 1]
            top_right = images[:, :, :n + 1, -n:]
            bottom_left = images[:, :, -n:, :n + 1]
            bottom_right = images[:, :, -n:, -n:]
            top_combined = torch.cat([top_left, top_right], axis=-2)
            # print(top_combined.shape)
            bottom_combined = torch.cat([bottom_left, bottom_right], axis=-2)
            # print(bottom_combined.shape)
            all_together = torch.cat([top_combined, bottom_combined], axis=-3)
            return all_together


class StudentNetwork_noRelu(nn.Module):
    def __init__(self):
        super(StudentNetwork_noRelu, self).__init__()
        self.conv1 = FFT_Conv_Layer(imgSize=224, inCs=3, outCs=32, imagDim=2, filtSize=3, cuda=True)
        self.conv2 = FFT_Conv_Layer(imgSize=113, inCs=32, outCs=64, imagDim=2, filtSize=3, cuda=True)
        self.conv3 = FFT_Conv_Layer(imgSize=55, inCs=64, outCs=128, imagDim=2, filtSize=3, cuda=True)
        self.conv4 = FFT_Conv_Layer(imgSize=27, inCs=128, outCs=256, imagDim=2, filtSize=3, cuda=True)
        self.conv5 = FFT_Conv_Layer(imgSize=13, inCs=256, outCs=256, imagDim=2, filtSize=3, cuda=True)
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
        self.max_113 = spectral_pool_layer(113)
        self.max_55 = spectral_pool_layer(55)
        self.max_27 = spectral_pool_layer(27)
        self.max_13 = spectral_pool_layer(13)

    def forward(self, x):
        forw = self.max_113(self.conv1(x))
        forw = self.max_55(self.conv2(forw))
        forw = self.max_27(self.conv3(forw))
        forw = self.max_13(self.conv4(forw))
        forw = self.conv5(forw)

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
