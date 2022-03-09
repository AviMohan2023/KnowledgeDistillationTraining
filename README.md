# Knowledge Distillation Circumvents Nonlinearity for Optical Convolutional Neural Networks

**Keywords: Python, Pytorch, Deep Learning, CNN, ONN**

# Introduction:
This repository contains the code for the paper "Knowledge Distillation Circumvents Nonlinearity for Optical Convolutional Neural Networks", which is available [here](https://arxiv.org/pdf/2102.13323.pdf), published in Applied Optics.

<img height="200" src="/Figures/fig1.png" title="Illustration of KD training." width="200"/>
<img height="200" src="/Figures/fig2.png" title="Nonlinear CNN (top) and SCLC" width="400"/>
# Abstract:
In recent years, Convolutional Neural Networks (CNNs) have enabled ubiquitous image processing applications. As such, CNNs require fast runtime (forward propagation) to process high-resolution visual streams in real time. This is still a challenging task even with state-of-the-art graphics and tensor processing units. The bottleneck in computational efficiency primarily occurs in the convolutional layers. Performing operations in the Fourier domain is a promising way to accelerate forward propagation since it transforms convolutions into elementwise multiplications, which are considerably faster to compute for large kernels. Furthermore, such computation could be implemented using an optical 4f system with orders of magnitude faster operation. However, a major challenge in using this spectral approach, as well as in an optical implementation of CNNs, is the inclusion of a nonlinearity between each convolutional layer, without which CNN performance drops dramatically. Here, we propose a Spectral CNN Linear Counterpart (SCLC) network architecture and develop a Knowledge Distillation (KD) approach to circumvent the need for a nonlinearity and successfully train such networks. While the KD approach is known in machine learning as an effective process for network pruning, we adapt the approach to transfer the knowledge from a nonlinear network (teacher) to a linear counterpart (student). We show that the KD approach can achieve performance that easily surpasses the standard linear version of a CNN and could approach the performance of the nonlinear network. Our simulations show that the possibility of increasing the resolution of the input image allows our proposed 4f optical linear network to perform more efficiently than a nonlinear network with the same accuracy on two fundamental image processing tasks: (i) object classification and (ii) semantic segmentation.

# Example:
Our training example are showed in jupyter notebook.

# Requirements

Tensorflow 1.14.0

Python 3

scikit-learn 0.21.2

matplotlib 3.1.0

numpy 1.16.4

# DATASET: 

**Object Classfication**: 

Cats Vs. Dogs

Cifar-10

HIGH-10, available here: [train](https://drive.google.com/file/d/1qS1E9_sm6EIzS3iY-CHcm0p1LUB3HOg7/view?usp=sharing) and [test](
https://drive.google.com/file/d/1w-qAPoJwiugqfbxWrbCnDSKm2YT29U_x/view?usp=sharing) 

**Oject Segementation**: 

VOC2012

Car Segementation

Face Recognition

# Citation:
Please cite the Knowledge Distillation Circumvents Nonlinearity for Optical Convolutional Neural Networks when you use this code.

`@book{xiang2021knowledge,
  title={Knowledge Distillation Circumvents Nonlinearity for Optical Convolutional Neural Networks},
  author={Xiang, Jinlin},
  year={2021},
  publisher={University of Washington}
}
`


