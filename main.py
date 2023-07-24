'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import random
import numpy as np

from models import *
from loader import Loader, Loader2
from utils import progress_bar
import time ### 

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

txt_name = "main_best_6_b50_balancedClass_to_c3"
checkpoint_name = "checkpoint_6_b50_balancedClass_to_c3"
total_cycle = 6
total_budget = 300
cycle_budget = total_budget//total_cycle


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "1"  # Set the GPUs 3 to use

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = Loader(is_train=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4) ### 2 -> 8

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = ResNet18()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# Training
def train(net, criterion, optimizer, epoch, trainloader):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(net, criterion, epoch, cycle):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(f'{checkpoint_name}'): #####
            os.mkdir(f'{checkpoint_name}') #####
        torch.save(state, f'./{checkpoint_name}/main_{cycle}.pth') #####
        best_acc = acc

# class-balanced sampling (pseudo labeling)
def get_plabels(net, samples, cycle):
    # dictionary with 10 keys as class labels
    class_dict = {}
    [class_dict.setdefault(x,[]) for x in range(10)]

    sub5k = Loader2(is_train=False,  transform=transform_test, path_list=samples)
    ploader = torch.utils.data.DataLoader(sub5k, batch_size=1, shuffle=False, num_workers=4) ### 2 -> 8

    # overflow goes into remaining
    remaining = []
    net.eval()
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(ploader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            if len(class_dict[predicted.item()]) < 100:
                class_dict[predicted.item()].append(samples[idx])
            else:
                remaining.append(samples[idx])
            progress_bar(idx, len(ploader))

    sample1k = []
    for items in class_dict.values():
        if len(items) == 100:
            sample1k.extend(items)
        else:
            # supplement samples from remaining 
            sample1k.extend(items)
            add = 100 - len(items)
            sample1k.extend(remaining[:add])
            remaining = remaining[add:]
    
    return sample1k

# confidence sampling (pseudo labeling)
## return 1k samples w/ lowest top1 score
def get_plabels2(net, samples, cycle):
    # dictionary with 10 keys as class labels
    class_dict = {}
    [class_dict.setdefault(x,[]) for x in range(10)]

    sample1k = []
    sub5k = Loader2(is_train=False,  transform=transform_test, path_list=samples)
    ploader = torch.utils.data.DataLoader(sub5k, batch_size=1, shuffle=False, num_workers=4) ### 2 -> 8

    top1_scores = []
    net.eval()
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(ploader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            scores, predicted = outputs.max(1)
            # save top1 confidence score 
            outputs = F.normalize(outputs, dim=1)
            probs = F.softmax(outputs, dim=1)
            top1_scores.append(probs[0][predicted.item()])
            progress_bar(idx, len(ploader))
    top1_scores = torch.tensor(top1_scores).cpu().numpy() ### 추가
    idx = np.argsort(top1_scores)
    samples = np.array(samples)
    return samples[idx[:cycle_budget]] ##### 1000

# entropy sampling
def get_plabels3(net, samples, cycle):
    sample1k = []
    sub5k = Loader2(is_train=False,  transform=transform_test, path_list=samples)
    ploader = torch.utils.data.DataLoader(sub5k, batch_size=1, shuffle=False, num_workers=4) ### 2 -> 8

    top1_scores = []
    net.eval()
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(ploader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            e = -1.0 * torch.sum(F.softmax(outputs, dim=1) * F.log_softmax(outputs, dim=1), dim=1)
            top1_scores.append(e.view(e.size(0)))
            progress_bar(idx, len(ploader))
    idx = np.argsort(top1_scores)
    samples = np.array(samples)
    return samples[idx[-1000:]]

def get_classdist(samples):
    class_dist = np.zeros(10)
    for sample in samples:
        label = int(sample.split('/')[-2])
        class_dist[label] += 1
    return class_dist

def get_balanced_class(samples, budget=1000, class_num=10): ## 실험용
    sample1k = []
    class_dist = np.zeros(10)
    max = budget//class_num
    for sample in samples:
        label = int(sample.split('/')[-2])
        if class_dist[label] >= max:
            continue
        else :
            class_dist[label] += 1
            samples.remove(sample)
            sample1k.append(sample)
    if len(sample1k) < budget: # 샘플 부족하면 아무거나 추가
        for sample in samples:
            label = int(sample.split('/')[-2])
            class_dist[label] += 1
            sample1k.append(sample)
            if len(sample1k) >= budget:
                break
    print('sample distribution = ', class_dist)
    sample1k = np.array(sample1k)
    return sample1k


if __name__ == '__main__':
    start = time.time() ###

    labeled = []
        
    CYCLES = total_cycle #####
    for cycle in range(CYCLES):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[160])

        best_acc = 0
        print('Cycle ', cycle)

        # open 5k batch (sorted low->high)
        with open(f'./loss_6/batch_{cycle}.txt', 'r') as f: #####
            samples = f.readlines()
            
        if cycle > 0:
            print('>> Getting previous checkpoint')
            # prevnet = ResNet18().to(device)
            # prevnet = torch.nn.DataParallel(prevnet)
            checkpoint = torch.load(f'./{checkpoint_name}/main_{cycle-1}.pth') #####
            net.load_state_dict(checkpoint['net'])

            # sampling
            sample1k = get_plabels2(net, samples, cycle)
        else:
            # first iteration: sample 1k at even intervals
            samples = np.array(samples)
            sample1k = samples[[j*5 for j in range(cycle_budget)]] ##### 1000
            # sample1k = get_balanced_class(samples, cycle_budget, 10) ##### 실험용
        # add 1k samples to labeled set
        labeled.extend(sample1k)
        print(f'>> Labeled length: {len(labeled)}')
        trainset = Loader2(is_train=True, transform=transform_train, path_list=labeled)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4) ### 2 -> 8

        for epoch in range(200): ###
            train(net, criterion, optimizer, epoch, trainloader)
            test(net, criterion, epoch, cycle)
            scheduler.step()
        with open(f'./{txt_name}.txt', 'a') as f: #####
            f.write(str(cycle) + ' ' + str(best_acc)+'\n')
    
    end = time.time()

    print(f"...it takes {end - start} sec")
