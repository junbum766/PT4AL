import torch
import torchvision
from PIL import Image
import os


class save_dataset(torch.utils.data.Dataset):

  def __init__(self, dataset, split='train'):
    self.dataset = dataset
    self.split = split

  def __getitem__(self, i):
      x, y = self.dataset[i]
      path = '/data/junbeom/data/cifar10/PT4AL/'+self.split+'/'+str(y)+'/'+str(i)+'.png'

      if not os.path.isdir('/data/junbeom/data/cifar10/PT4AL/'+self.split+'/'+str(y)):
          os.mkdir('/data/junbeom/data/cifar10/PT4AL/'+self.split+'/'+str(y))

      x.save(path)

  def __len__(self):
    return len(self.dataset)

trainset = torchvision.datasets.CIFAR10(root='/data/junbeom/data/cifar10', train=True, download=False, transform=None)

testset = torchvision.datasets.CIFAR10(root='/data/junbeom/data/cifar10', train=False, download=False, transform=None)

train_dataset = save_dataset(trainset, split='train')
test_dataset = save_dataset(testset, split='test')

if not os.path.isdir('/data/junbeom/data/cifar10/PT4AL'):
    os.mkdir('/data/junbeom/data/cifar10/PT4AL')

if not os.path.isdir('/data/junbeom/data/cifar10/PT4AL/train'):
    os.mkdir('/data/junbeom/data/cifar10/PT4AL/train')

if not os.path.isdir('/data/junbeom/data/cifar10/PT4AL/test'):
    os.mkdir('/data/junbeom/data/cifar10/PT4AL/test')

for idx, i in enumerate(train_dataset):
    train_dataset[idx]
    print(idx)

for idx, i in enumerate(test_dataset):
    test_dataset[idx]
    print(idx)
