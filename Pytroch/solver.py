import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from tqdm import tqdm
from network_single import *
from dataload import dataload
import pickle
import argparse
import time
import itertools
from copy import deepcopy
import os
from datetime import datetime


class teacher_solver():

    def __init__(self, train_loader, test_loader, model, criterion, student_optimizer,
                 student_lr_scheduler,
                 epochs, model_path, model_name):

        self.model_path = model_path
        self.model_name = model_name
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.student_optimizer = student_optimizer
        self.student_lr_scheduler = student_lr_scheduler
        self.epochs = epochs
        self.criterion = criterion
        self.step = 0

    def train(self):
        val_loss = None
        print('epochs', self.epochs)
        for epoch in range(self.epochs):
            print("Start Training...")
            self.val_predictions = []
            self.val_gts = []
            start = datetime.now()
            tr_stu_avg_loss = self.train_loop()
            val_stu_avg_loss, testaccuracy = self.validate()
            print('-' * 50)
            print('Summary: Epoch {0} | Time {1}s'.format(epoch, datetime.now() - start))
            print('Train | Loss {0:.4f}'.format(tr_stu_avg_loss))
            print('Validate | Loss {0:.4f}'.format(val_stu_avg_loss))
            print('Validate | Accuracy {0:.4f}'.format(testaccuracy))
            # load the model
            if val_loss is None or val_stu_avg_loss < val_loss:
                val_loss = val_stu_avg_loss
                torch.save(self.model.state_dict(), self.model_path + self.model_name)
                best_model = epoch
            print('best_model is on epoch:', best_model)


    def train_loop(self):
        self.model.train()
        running_loss = 0
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
            logps = self.model.forward(inputs)
            loss = self.criterion(logps, labels)
            loss.backward()
            self.student_optimizer.step()
            running_loss += loss.item()
        traininglosses = running_loss / len(self.train_loader)
        return traininglosses

    def validate(self):
        self.model.eval()
        test_loss = 0
        accuracy = 0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
                logps = self.model.forward(inputs)
                batch_loss = criterion(logps, labels)
                test_loss += batch_loss.item()
                # Calculate accuracy
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()


        testinglosses = test_loss / len(self.test_loader)
        testaccuracy = accuracy / len(self.test_loader)
        return testinglosses, testaccuracy


class student_solver():

    def __init__(self, train_loader, test_loader, model, teacher_model, criterion, student_optimizer,
                 student_lr_scheduler,
                 epochs, model_path, model_name,temperatures=10, alphas=0.5, learning_rates=0.0001,
                 learning_rate_decays=0.95, weight_decays=1e-5, momentums= 0.9, dropout_probabilities = (0,0)):

        self.model_path = model_path
        self.model_name = model_name
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.teacher_model = teacher_model
        self.student_optimizer = student_optimizer
        self.student_lr_scheduler = student_lr_scheduler
        self.epochs = epochs
        self.criterion = criterion
        self.step = 0
        self.T = temperatures
        self.alphas = alphas
        self.dropout_input = dropout_probabilities[0]
        self.dropout_hidden = dropout_probabilities[1]
        self.lr_decay = learning_rate_decays
        self.weight_decay = weight_decays
        self.momentum = momentums
        self.lr = learning_rates

        reproducibilitySeed()

    def train(self):
        """
        Trains teacher on given hyperparameters for given number of epochs; Pass val_loader=None when not required to validate for every epoch
        Return: List of training loss, accuracy for each update calculated only on the batch; List of validation loss, accuracy for each epoch
        """
        self.model.dropout_input = self.dropout_input
        self.model.dropout_hidden = self.dropout_hidden
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr,
                              momentum=self.momentum, weight_decay=self.weight_decay)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.lr_decay)

        for epoch in range(self.epochs):
            print("Start KD Training...")
            lr_scheduler.step()
            start = datetime.now()
            tr_stu_avg_loss = self.train_loop()
            val_stu_avg_loss, testaccuracy = self.validate()
            print('-' * 50)
            print('Summary: Epoch {0} | Time {1}s'.format(epoch, datetime.now() - start))
            print('Train | Loss {0:.4f}'.format(tr_stu_avg_loss))
            print('Validate | Loss {0:.4f}'.format(val_stu_avg_loss))
            print('Validate | Accuracy {0:.4f}'.format(testaccuracy))
            # load the model
            if val_loss is None or val_stu_avg_loss < val_loss:
                val_loss = val_stu_avg_loss
                torch.save(self.model.state_dict(), self.model_path + self.model_name)
                best_model = epoch
            print('best_model is on epoch:', best_model)

    def train_loop(self):
        print_every = 1000
        for i, data in enumerate(self.train_loader, 0):
            X, y = data
            X, y = X.to('cuda'), y.to('cuda')
            optimizer.zero_grad()
            teacher_pred = None
            if (self.alphas > 0):
                with torch.no_grad():
                    teacher_pred = self.teacher_model(X)
            student_pred = self.model(X)
            loss = self.studentLossFn(teacher_pred, student_pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 20)
            optimizer.step()
            # accuracy = float(torch.sum(torch.argmax(student_pred, dim=1) == y).item()) / y.shape[0]
            if i % print_every == 0:
                loss, acc = self.validate()
                print('train loss: %.3f, train loss: %.3f' %(loss, acc))
        return loss

    def validate(self):
        loss, val_acc = self.getLossAccuracyOnDataset(self.model, self.test_loader, 'cuda')
        return loss, val_acc


    def getLossAccuracyOnDataset(self, network, dataset_loader, fast_device, criterion=None):
        """
        Returns (loss, accuracy) of network on given dataset
        """
        network.is_training = False
        accuracy = 0.0
        loss = 0.0
        dataset_size = 0
        for j, D in enumerate(dataset_loader, 0):
            X, y = D
            X = X.to(fast_device)
            y = y.to(fast_device)
            with torch.no_grad():
                pred = network(X)
                if criterion is not None:
                    loss += criterion(pred, y) * y.shape[0]
                accuracy += torch.sum(torch.argmax(pred, dim=1) == y).item()
            dataset_size += y.shape[0]
        loss, accuracy = loss / dataset_size, accuracy / dataset_size
        network.is_training = True
        return loss, accuracy

    def studentLossFn(self, teacher_pred, student_pred, y):
        """
        Loss function for student network: Loss = alpha * (distillation loss with soft-target) + (1 - alpha) * (cross-entropy loss with true label)
        Return: loss
        """
        T = self.T
        alpha = self.alphas
        if (alpha > 0):
            loss = F.kl_div(F.log_softmax(student_pred / T, dim=1), F.softmax(teacher_pred / T, dim=1),
                            reduction='batchmean') * (T ** 2) * alpha + F.cross_entropy(student_pred, y) * (
                           1 - alpha)
        else:
            loss = F.cross_entropy(student_pred, y)
        return loss


def reproducibilitySeed(use_gpu=True):
    """
    Ensure reproducibility of results; Seeds to 0
    """
    torch_init_seed = 0
    torch.manual_seed(torch_init_seed)
    numpy_init_seed = 0
    np.random.seed(numpy_init_seed)
    if use_gpu:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    # preparation
    train_loader, test_loader = dataload('MNIST', bs=10)
    teacher_model = Teacher_Network(in_channels=1).cuda()
    student_model = StudentNetwork_noRelu(in_channels=1).cuda()
    model_path = 'models/'
    teacher_model_name ='teacher.pth'
    student_model_name = 'student.pth'


    # step 1: train teacher:
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(teacher_model.parameters(), lr=0.003)
    epochs = 50
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.95)

    solver = teacher_solver(train_loader, test_loader, teacher_model, criterion, optimizer,
                 lr_scheduler,
                 epochs, model_path, teacher_model_name)
    solver.train()

    #step 2: load teacher model:

    teacher_model.load_state_dict(torch.load(model_path+teacher_model_name))
    print("init weight from {}".format(model_path))
    print(sum([param.nelement() * param.element_size() for param in teacher_model.parameters()]))

    # step 3: training student model
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(student_model.parameters(), lr=0.003)
    epochs = 50
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.95)

    solver = student_solver(train_loader, test_loader, student_model, teacher_model, criterion, optimizer,
                                          lr_scheduler,
                                          epochs, model_path, student_model_name)
    solver.train()
