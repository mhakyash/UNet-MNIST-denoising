import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim


#=====================================================================

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    

std_added_noise = 20

data_transform = {
        'transformation_adding_noise': transforms.Compose([                                                        
        transforms.ToTensor(),
        AddGaussianNoise(std = std_added_noise/255),
        transforms.Resize(64),
        transforms.ConvertImageDtype(torch.float)
        ]),
        'transformation_original': transforms.Compose([
        transforms.ToTensor(),                                           
        transforms.Resize(64),
        transforms.ConvertImageDtype(torch.float)
        ])
                    }

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform = data_transform['transformation_original'])
mnist_trainset_noisy = datasets.MNIST(root='./data', train=True, download=True, transform = data_transform['transformation_adding_noise'])

mnist_trainset_input = []
for input_tensor,_ in mnist_trainset_noisy:
    mnist_trainset_input.append(input_tensor)

mnist_trainset_output = []
for input_tensor,_ in mnist_trainset:
    mnist_trainset_output.append(input_tensor)


train_loader_input = DataLoader(
    mnist_trainset_input,
    batch_size=64,
    num_workers=2,
    shuffle=False)

train_loader_output = DataLoader(
    mnist_trainset_output,
    batch_size=64,
    num_workers=2,
    shuffle=False)


#=====================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class double_conv(nn.Module):

    def __init__(self, input_channels, output_channels):
        super(double_conv, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size = 3, padding='same')
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size = 3, padding='same')
        self.relu_activation = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.conv1(self.relu_activation(x))
        x = self.conv2(self.relu_activation(x))

        return x


class traspose_conv(nn.Module):

    def __init__(self, num_of_channels):
        super(traspose_conv, self).__init__()
        self.trasnpose_conv = nn.ConvTranspose2d(num_of_channels, int(num_of_channels / 2), kernel_size = 2, stride = 2)

    def forward(self, x):

        x = self.trasnpose_conv(x)

        return x

class double_decoder_conv(nn.Module):

    def __init__(self, input_channels1, output_channels1, output_channels2):
        super(double_decoder_conv, self).__init__()
        self.conv1 = nn.Conv2d(input_channels1, output_channels1, kernel_size = 3, padding='same')
        self.conv2 = nn.Conv2d(output_channels1, output_channels2, kernel_size = 3, padding='same')
        self.relu_activation = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.conv1(self.relu_activation(x))
        x = self.conv2(self.relu_activation(x))

        return x

class UNet(nn.Module):

    def __init__(self):
        super(UNet, self).__init__()

        self.double_conv1 = double_conv(1 , 64)
        self.double_conv2 = double_conv(64, 128)
        self.double_conv3 = double_conv(128, 256)
        self.double_conv4 = double_conv(256, 512)
        self.double_conv5 = double_conv(512, 1024)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
           
        self.traspose_conv1 = traspose_conv(1024)         
        self.traspose_conv2 = traspose_conv(512)
        self.traspose_conv3 = traspose_conv(256)
        self.traspose_conv4 = traspose_conv(128)

        self.double_decoder_conv1 = double_decoder_conv(1024 , 512, 512)
        self.double_decoder_conv2 = double_decoder_conv(512, 256, 256)
        self.double_decoder_conv3 = double_decoder_conv(256, 128, 128)
        self.double_decoder_conv4 = double_decoder_conv(128, 64, 64)

        self.final_conv = nn.Conv2d(64, 1, kernel_size=1, padding = 'same')


    def forward(self, x):

        conv_output1 = self.double_conv1(x)
        conv_output2 = self.double_conv2(self.maxpool(conv_output1))
        conv_output3 = self.double_conv3(self.maxpool(conv_output2))
        conv_output4 = self.double_conv4(self.maxpool(conv_output3))
        x = self.double_conv5(self.maxpool(conv_output4))
        
        x = self.traspose_conv1(x)
        x = torch.cat([x, conv_output4], dim=1)
        x = self.double_decoder_conv1(x)
        
        x = self.traspose_conv2(x)
        x = torch.cat([x, conv_output3], dim=1)
        x = self.double_decoder_conv2(x)
        
        x = self.traspose_conv3(x)
        x = torch.cat([x, conv_output2], dim=1)
        x = self.double_decoder_conv3(x)
        
        x = self.traspose_conv4(x)
        x = torch.cat([x, conv_output1], dim=1)
        x = self.double_decoder_conv4(x)
        

        x = self.final_conv(x)
        
        return x

#=====================================================================

model = UNet()
model = model.to(device)

#=====================================================================

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


training_loss = []
for epoch in range(5):  
    i = 0

    for input_data, output_data in zip(train_loader_input, train_loader_output):

        input_data = input_data.to(device)
        output_data = output_data.to(device)

        optimizer.zero_grad()

        outputs = model(input_data)
        loss = criterion(outputs, output_data)
        loss.backward()
        optimizer.step()
    
        
        if i % 100 == 99: 
            print(f'#Epoch: {epoch + 1}, #Batch {i + 1:5d}, Training Loss: {loss.item():.6f}')
            training_loss.append(loss.item())
        i += 1

#=====================================================================

plt.plot(training_loss)
plt.title('Learning Curve')
plt.xlabel('#iteration')
plt.ylabel('Training Loss')
plt.show()

#=====================================================================


mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform = data_transform['transformation_original'])
mnist_testset_noisy = datasets.MNIST(root='./data', train=False, download=True, transform = data_transform['transformation_adding_noise'])


sample_number = 100

original_image = mnist_testset[sample_number][0][0]
noisy_image = mnist_testset_noisy[sample_number][0][0]
output_denoised = model(mnist_testset_noisy[sample_number][0].unsqueeze(dim = 0).to(device)).cpu().detach().numpy().reshape((64,64))

#=====================================================================

f, axes = plt.subplots(1, 3)

axes[0].imshow(original_image, cmap = 'gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(noisy_image, cmap = 'gray')
axes[1].set_title('Noisy Image')
axes[1].axis('off')

axes[2].imshow(output_denoised, cmap = 'gray')
axes[2].set_title('Network Output')
axes[2].axis('off')

plt.show()