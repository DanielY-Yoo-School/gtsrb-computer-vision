
# coding: utf-8

# In[1]:


import json
from glob import glob
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler

import torchvision
from torchvision import transforms, datasets

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import time

from sklearn.model_selection import train_test_split


# In[2]:


training_path = "./Data/Training"
test_path = "./Data/Test"
sign_label_path = "./Data/sign_labels.csv"
batch_size = 64
random_seed = 42
class_count = 43  # 43 classes in GTSRB
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')


# In[3]:


def showLabelDistribution(dataset, dataloader=None):
    labelDict = dict.fromkeys(range(class_count), 0)
    print(labelDict)

    if(dataloader == None):
        for i in range(len(dataset)):
            label = int(dataset[i][1])
            labelDict[label] = labelDict[label] + 1
    else:
        for _, (images, labels) in enumerate(dataloader):
            for label in labels:
                label = int(label)
                labelDict[label] += 1

    print(labelDict)
    plt.bar(labelDict.keys(), labelDict.values(), width=1.0, color='g')
    plt.show()


# In[5]:


# The German Traffic Sign Recognition Benchmark
#
# sample code for reading the traffic sign images and the
# corresponding labels
#
# example:
#
# trainImages, trainLabels = readTrafficSigns('GTSRB/Training')
# print len(trainLabels), len(trainImages)
# plt.imshow(trainImages[42])
# plt.show()
#
# have fun, Christian
# Source: http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset#Downloads

# function for reading the images
# arguments: path to the traffic sign data, for example './GTSRB/Training'
# returns: list of images, list of corresponding labels
def readTrafficSigns(rootpath):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''

    images = []  # images
    labels = []  # corresponding labels
    # loop over all 42 classes
    for c in range(0, class_count):
        # subdirectory for class
        prefix = rootpath + '/' + format(c, '05d') + '/'
        gtFile = open(prefix + 'GT-' + format(c, '05d') +
                      '.csv')  # annotations file
        # csv parser for annotations file
        gtReader = csv.reader(gtFile, delimiter=';')
        next(gtReader)  # skip header
        # loop over all images in current annotations file
        for row in gtReader:
            # the 1th column is the filename
            images.append(plt.imread(prefix + row[0]))
            labels.append(row[7])  # the 8th column is the label
        gtFile.close()
    return images, labels


# In[6]:


class TrafficSignDataset(data.Dataset):
    def __init__(self, images, labels):
        self.images, self.labels = images, labels
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            # Samples are randomly perturbed in position ([-2,2] pixels),
            # in scale ([.9,1.1] ratio) and rotation ([-15,+15] degrees).
            # Source: http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf
            transforms.RandomApply([
                transforms.RandomRotation(15, resample=Image.BICUBIC),
                transforms.RandomAffine(0, translate=(
                    0.2, 0.2), resample=Image.BICUBIC),
                transforms.RandomAffine(0, shear=20, resample=Image.BICUBIC),
                transforms.RandomAffine(0, scale=(0.9, 1.1),
                                        resample=Image.BICUBIC)
            ]),
            transforms.ToTensor(),
            transforms.Normalize((0.3337, 0.3064, 0.3171),
                                 (0.2672, 0.2564, 0.2629))
        ])

    def __getitem__(self, index):
        image = self.transforms(self.images[index])
        labels = np.asarray(self.labels[index])
        labels = torch.from_numpy(labels.astype('int'))
        return image, labels

    def __len__(self):
        return len(self.images)


# In[7]:


# Assign Random Seed
np.random.seed(random_seed)


# In[8]:


# Loading training data
images, labels = readTrafficSigns(training_path)

# Splitting the data to 8:2 ratios
train_images, val_images, train_labels, val_labels = train_test_split(
    images, labels, test_size=0.2, random_state=1)


# In[9]:


training_dataset = TrafficSignDataset(train_images, train_labels)
validation_dataset = TrafficSignDataset(val_images, val_labels)


# In[10]:


print(len(training_dataset))
print(len(validation_dataset))


# In[11]:


# Check the distribution of labels
showLabelDistribution(training_dataset)
showLabelDistribution(validation_dataset)


# In[31]:


# Even out the class distribution for training set
classidx = np.bincount(training_dataset.labels)
weights = 1 / np.array([classidx[int(y)] for y in training_dataset.labels])
sampler = WeightedRandomSampler(weights, len(weights) * class_count)


# In[32]:


# Splitting the Dataset into Training Dataloader and the Validation Dataloader
train_loader = torch.utils.data.DataLoader(
    training_dataset, batch_size=batch_size, sampler=sampler, num_workers=4, drop_last=True)
validation_loader = torch.utils.data.DataLoader(
    validation_dataset, batch_size=batch_size, num_workers=4)


# In[14]:


# Make the distribution more even for better training results
showLabelDistribution(_, train_loader)
# showLabelDistribution(_, validation_loader)


# In[15]:


for i, (images, labels) in enumerate(train_loader):
    print(images.shape)
    print(labels)
    break


# In[33]:


class TrafficSignNet(nn.Module):
    # Batch shape is (3,32,32)
    def __init__(self):
        super(TrafficSignNet, self).__init__()

        # prevent overfitting by dropping randomly
        self.dropout = nn.Dropout(p=0.5)

        # convolutional 100 maps of 42x42 neurons 7x7
        self.conv1 = nn.Conv2d(3, 100, 7)

        # max pooling 100 maps of 21x21 neurons 2x2
        self.batch1 = nn.BatchNorm2d(100)
        self.pool = nn.MaxPool2d(2)

        # convolutional 150 maps of 18x18 neurons 4x4
        self.conv2 = nn.Conv2d(100, 150, 4)

        # max pooling 150 maps of 9x9 neurons 2x2
        self.batch2 = nn.BatchNorm2d(150)
        # use self.pool; Same pool kernal

        # convolutional 250 maps of 6x6 neurons 4x4
        self.conv3 = nn.Conv2d(150, 250, 4)

        # max pooling 250 maps of 3x3 neurons 2x2
        self.batch3 = nn.BatchNorm2d(250)
        # use self.pool; Same pool kernal

        # fully connected 300 neurons 1x1
        self.fc1 = nn.Linear(250 * 1 * 1, 300)
        self.batch4 = nn.BatchNorm1d(300)

        # fully connected 43 neurons 1x1
        self.fc2 = nn.Linear(300, 43)

    def forward(self, x):
        # convolutional 100 maps of 42x42 neurons 7x7
        x = self.conv1(x)

        # ELU activation function
        x = F.elu(x)

        # max pooling 100 maps of 21x21 neurons 2x2
        x = self.pool(x)
        x = self.batch1(x)

        # prevent overfitting by dropping randomly
        x = self.dropout(x)

        # convolutional 150 maps of 18x18 neurons 4x4
        x = self.conv2(x)

        # ELU activation function
        x = F.elu(x)

        # max pooling 150 maps of 9x9 neurons 2x2
        x = self.pool(x)
        x = self.batch2(x)

        # prevent overfitting by dropping randomly
        x = self.dropout(x)

        # convolutional 250 maps of 6x6 neurons 4x4
        x = self.conv3(x)

        # ELU activation function
        x = F.elu(x)

        # max pooling 250 maps of 3x3 neurons 2x2
        x = self.pool(x)
        x = self.batch3(x)

        # prevent overfitting by dropping randomly
        x = self.dropout(x)

        # fully connected 300 neurons 1x1
        x = x.view(-1, 250 * 1 * 1)
        x = self.fc1(x)

        # ELU activation function
        x = F.elu(x)
        # prevent overfitting by dropping randomly
        x = self.batch4(x)
        x = self.dropout(x)

        # fully connected 43 neurons 1x1
        x = self.fc2(x)

        return x


# In[34]:


print(device)


# In[35]:


net = TrafficSignNet()
net.to(device)


# In[36]:


num_epochs = 100
params = [p for p in net.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params, lr=0.0001, momentum=0.9, weight_decay=0.0005)
loss_function = nn.CrossEntropyLoss()

correct = 0
total = 0

consecutive_loss_increase = 0
prev_loss = np.Inf
running_loss = 0

train_losses = []
train_nums = []

val_losses = []
val_nums = []

training_accuracy_epoch = dict.fromkeys(range(100), [])
running_losses_epoch = dict.fromkeys(range(100), [])
training_losses_epoch = dict.fromkeys(range(100), [])
val_losses_epoch = dict.fromkeys(range(100), [])


# In[ ]:


for epoch in range(num_epochs):
    t0 = time.time()
    # Training
    print("Epoch %d: " % (epoch + 1))
    net.train()
    t1 = time.time()
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # forward + backward + optimize
        loss = loss_function(net(images), labels)
        loss.backward()
        optimizer.step()

        # zero the parameter gradients
        optimizer.zero_grad()

        train_losses.append(loss.item())
        train_nums.append(len(images))

        running_loss += loss.item()

        if i % 2000 == 1999:
            running_losses_epoch[epoch].append(running_loss / 2000)
            print('[%d, %5d] running loss: %.3f\t took: %.6f seconds' %
                  (epoch + 1, i + 1, running_loss / 2000, time.time() - t1))
            running_loss = 0.0
            t1 = time.time()

    training_loss = np.sum(np.multiply(
        train_losses, train_nums)) / np.sum(train_nums)
    training_losses_epoch[epoch].append(training_loss)
    print('[%d] training loss: %.3f\t' %
          (epoch + 1, training_loss))

    # Evaluate
    net.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(validation_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)

            loss = loss_function(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            val_losses.append(loss.item())
            val_nums.append(len(images))

            if i % 10 == 9:
                validation_accuracy = np.sum(correct) / total
                training_accuracy_epoch[epoch].append(validation_accuracy)
                print('[%d, %5d] Validation Accuracy: %.3f \t' %
                      (epoch + 1, i + 1, validation_accuracy))

    val_losses_epoch[epoch].append(new_loss)
    new_loss = np.sum(np.multiply(val_losses, val_nums)) / np.sum(val_nums)

    print('Validation Loss: %.3f \t' % (new_loss))

    if new_loss <= prev_loss:
        # save progress each valid epoch
        torch.save(net.state_dict(), './checkpoint/' + str(epoch) + '.pth')

        prev_loss = new_loss
        consecutive_loss_increase = 0
    else:
        consecutive_loss_increase += 1

        if(consecutive_loss_increase > 9):
            print("Ending training due to consecutive non-decrease of loss")

    print('Epoch %d: took: %.3f seconds' % (epoch + 1, time.time() - t0))

torch.save(net.state_dict(), './final_model.pth')


# In[39]:


# Check dataloaders have about the same sizes
print(len(train_loader) * batch_size)
print(len(validation_loader) * batch_size)


# In[41]:


print("Training Accuracy per Epoch: ")
print(training_accuracy_epoch)


# In[42]:


print("Running Losses per Epoch: ")
print(running_losses_epoch)


# In[43]:


print("Training Losses per Epoch: ")
print(training_losses_epoch)


# In[44]:


print("Validation Losses per Epoch: ")
print(val_losses_epoch)


# In[72]:


# Visualizing
fig, ax = plt.subplots()

x_val_loss = np.arange(1, 100, 1)
y_val_loss = val_losses_epoch[99][1:]

ax.plot(x_val_loss, y_val_loss)

ax.set(xlabel='Epoch #', ylabel='Validation Losses',
       title='Validation Losses')
ax.grid()

fig.savefig("validation_loss.png")
plt.show()


# In[92]:


# Visualizing
fig, ax = plt.subplots()

x_val_acc = np.arange(1, 100, 1/12)
y_val_acc = training_accuracy_epoch[99][12:]

ax.plot(x_val_acc, y_val_acc)

ax.set(xlabel='Epoch #', ylabel='Validation Accuracy (%)',
       title='Validation Accuracy')
ax.grid()

fig.savefig("validation_accuracy.png")
plt.show()


# In[93]:


# Make Testset
test_images, test_labels = readTrafficSigns(test_path)
test_dataset = TrafficSignDataset(test_images, test_labels)

print(len(test_dataset))
showLabelDistribution(test_dataset)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, num_workers=4)


# In[95]:


len(test_loader)


# In[97]:


# Test against testset
net.eval()

correct_test = 0
total_test = 0
with torch.no_grad():
    for i, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)

        _, predicted = torch.max(outputs.data, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()

        if i % 50 == 49:
            test_accuracy = np.sum(correct_test) / total_test
            print('[%5d] Test Accuracy: %.3f \t' % (i + 1, test_accuracy))

print('Accuracy of the network on the 26640 test images: %.6f %%' % (
    100 * correct_test / total_test))


# In[ ]:
