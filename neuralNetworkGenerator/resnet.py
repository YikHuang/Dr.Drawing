# -*- coding: utf-8 -*-
"""
Created on Thu May  2 22:06:49 2019

@author: user
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 15:38:52 2019

@author: user
"""



import torch
import torchvision
from torchvision import transforms, utils
from torch.autograd import Variable


def conv3x3(in_channels, out_channels, stride = 1):
    
    return torch.nn.Conv2d(in_channels, out_channels,
                           kernel_size = 3, stride = stride,
                           padding = 1, bias = False)


#殘差塊
class ResidualBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        
        super(ResidualBlock, self).__init__()
        
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()
        
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample
        
        
    def forward(self, x):
        
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            residual = self.downsample(x)

        out = residual + out
        out = self.relu(out)
        
        
        return out



class ResNet(torch.nn.Module):
    
    def __init__(self, block, layers, num_classes = 102):
        
        super(ResNet, self).__init__()
        
        self.in_channels = 64
        
        self.conv = conv3x3(3, 64)
        self.bn = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU(inplace = True)
        
        
        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], 2)
        self.layer3 = self.make_layer(block, 256, layers[2], 2)
        self.layer4 = self.make_layer(block, 512, layers[3], 2)

        
        self.avg_pool = torch.nn.AvgPool2d(4)
        
        self.fc = torch.nn.Linear(512, num_classes)
        
        

    def make_layer(self, block, out_channels, blocks, stride = 1):

        downsample = None
        
        if((stride != 1) or (self.in_channels != out_channels)):
            
            downsample = torch.nn.Sequential(
                    conv3x3(self.in_channels, out_channels, stride),
                    torch.nn.BatchNorm2d(out_channels)
                )
        
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        
        self.in_channels = out_channels
        
        for i in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return torch.nn.Sequential(*layers)


    def forward(self, x):
        
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        
        
        out = self.avg_pool(out)
        
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        
        return out
  

batch_size = 100
epoch_n = 1
learning_rate = 0.1

#先把圖片資料讀出來
train_data = torchvision.datasets.ImageFolder("picture_train",
                                              transform = transforms.ToTensor())



train_loader = torch.utils.data.DataLoader(dataset = train_data,
                                           batch_size = batch_size,
                                           shuffle = True)


print(train_data[1000][1])


test_data = torchvision.datasets.ImageFolder("picture_test",
                                             transform = transforms.ToTensor())


test_loader = torch.utils.data.DataLoader(dataset = test_data,
                                          batch_size = batch_size,
                                          shuffle = False)


print(test_data.class_to_idx)




resnet = ResNet(ResidualBlock, [3,4,23,3])
resnet.cuda()
print(resnet)


loss_f = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet.parameters(), lr = learning_rate)


for epoch in range(1, epoch_n+1):
    for i ,(images, labels) in enumerate(train_loader):
        
        x = Variable(images.cuda(), requires_grad = False)
        y = Variable(labels.cuda(), requires_grad = False)

        output = resnet(x)
    
        loss = loss_f(output, y)

        if(i % 10 == 0):
            print("Epoch:{}, Step:{}, Loss:{}".format(epoch, i, loss))

        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()




    if epoch % 20 == 0:
        torch.save(resnet, "resnet\resnet" + epoch +".pkl")
        torch.save(resnet.state_dict(), "resnet\resnet_params"+ epoch +".pkl")



total = 0
accuracy = 0

for images, labels in test_loader:
    
    test_x = Variable(images.cuda(), requires_grad = False)

    test_output = resnet(test_x)
    y_pred = torch.max(test_output, 1)[1]
    
    print("預測")
    print(y_pred)
    print("答案")
    print(labels)
    
    total += labels.size(0)
    accuracy += (y_pred.cpu() == labels).sum()


correct = accuracy.data.numpy()
print("Accuracy : {} %".format(100 * correct / total))


