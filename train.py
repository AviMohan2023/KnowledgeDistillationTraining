import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from tqdm import tqdm
from network_MNIST import *

if __name__ == '__main__':
    '''data agumentaiton'''
    # traindir ='train/'
    # testdir = 'test/'
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                      std=[0.229, 0.224, 0.225])
    #
    # train_dataset = datasets.ImageFolder(
    #     traindir,
    #     transforms.Compose([
    #         transforms.Resize((256,256)),
    #         transforms.RandomResizedCrop(224),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize,
    #     ]))
    #
    # test_dataset = datasets.ImageFolder(
    #     testdir,
    #     transforms.Compose([
    #         transforms.Resize((224,224)),
    #         transforms.ToTensor(),
    #         normalize,
    #     ]))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root='../data', train=True, download=True,
                              transform=transform)
    test_dataset = datasets.MNIST(root='../data', train=False,
                              transform=transform)
    trainloader = torch.utils.data.DataLoader(train_dataset,batch_size=16, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_dataset,batch_size=16, shuffle=True)

    device = 'cuda'
    model = StudentNetwork_noRelu()
    # model = teacher_net
    criterion = nn.NLLLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.003)
    optimizer = optim.SGD(model.parameters(), lr=0.003)
    model.to(device)



    traininglosses = []
    testinglosses = []
    testaccuracy = []
    totalsteps = []
    epochs = 50
    steps = 0
    running_loss = 0
    print_every = 100
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            #print(inputs.shape)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in testloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                traininglosses.append(running_loss / print_every)
                testinglosses.append(test_loss / len(testloader))
                testaccuracy.append(accuracy / len(testloader))
                totalsteps.append(steps)
                print(f"Device {device}.."
                      f"Epoch {epoch + 1}/{epochs}.. "
                      f"Step {steps}.. "
                      f"Train loss: {running_loss / print_every:.3f}.. "
                      f"Test loss: {test_loss / len(testloader):.3f}.. "
                      f"Test accuracy: {accuracy / len(testloader):.3f}")
                running_loss = 0
                model.train()