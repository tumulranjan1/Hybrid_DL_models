import os
import glob
import torch
import shutil
import itertools
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from torch.nn import functional as F
from torchvision import datasets, models, transforms
import wandb
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report, plot_confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(torch.__version__)
print(torchvision.__version__)

normalizer = transforms.Normalize(mean=[0.5], std=[0.25])

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((244, 244)),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        normalizer
    ]),
    
    'validation': transforms.Compose([
        transforms.Resize((244, 244)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        normalizer
    ])
}

data_images = {
    'train': datasets.ImageFolder('./train', data_transforms['train']),
    'validation': datasets.ImageFolder('./test', data_transforms['validation'])
}

dataloaders = {
    'train': torch.utils.data.DataLoader(data_images['train'], batch_size=64, shuffle=True, num_workers=0),
    'validation': torch.utils.data.DataLoader(data_images['validation'], batch_size=64,shuffle=True,num_workers=0)
}

len(data_images['validation'])

training_loss = []
validation_loss = []
training_accuracy = []
validation_accuracy = []

def trained_model(model, criterion, optimizer, epochs, scaler=None):
    for epoch in range(epochs):
        
        print('Epoch:', str(epoch+1) + '/' + str(epochs))
        print('-'*10)
        
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train() #this trains the model
            else:
                model.eval() #this evaluates the model

            running_loss, running_corrects = 0.0, 0 
            all_preds, all_labels = [], []

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device) #convert inputs to cpu or cuda
                labels = labels.to(device) #convert labels to cpu or cuda
                outputs = model(inputs) #outputs is inputs being fed to the model
                loss = criterion(outputs, labels) #outputs are fed into the model

                if phase == 'train':
                    optimizer.zero_grad() #sets gradients to zero
                    loss.backward() #computes sum of gradients
                    optimizer.step() #preforms an optimization step

                _, preds = torch.max(outputs, 1) #max elements of outputs with output dimension of one
                running_loss += loss.item() * inputs.size(0) #loss multiplied by the first dimension of inputs
                running_corrects += torch.sum(preds == labels.data) #sum of all the correct predictions
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            epoch_loss = running_loss / len(data_images[phase]) #this is the epoch loss
            epoch_accuracy = running_corrects.double() / len(data_images[phase]) #this is the epoch accuracy

            print(phase, ' loss:', epoch_loss, 'epoch_accuracy:', epoch_accuracy)

            if phase == 'train':
                training_loss.append(epoch_loss)
                training_accuracy.append(epoch_accuracy.item())
            else:
                validation_loss.append(epoch_loss)
                validation_accuracy.append(epoch_accuracy.item())

            # Calculate confusion matrix
            cm = confusion_matrix(all_labels, all_preds)

            # Calculate precision, recall, and F1 score
            precision = precision_score(all_labels, all_preds, average='macro')
            recall = recall_score(all_labels, all_preds, average='macro')
            f1 = f1_score(all_labels, all_preds, average='macro')

            # Log metrics to Weights and Biases
            wandb.log({f'{phase}_loss': epoch_loss,
                       f'{phase}_accuracy': epoch_accuracy,
                    #    f'{phase}_confusion_matrix': wandb.plot.confusion_matrix(probs=None,
                    #                                                            y_true=all_labels,
                    #                                                            preds=all_preds,
                    #                                                            class_names=['class1', 'class2', 'class3']),
                       f'{phase}_precision': precision,
                       f'{phase}_recall': recall,
                       f'{phase}_f1_score': f1})

            # Plot confusion matrix with numbers
            plt.figure(figsize=(8, 6))
            plot_confusion_matrix(cm, classes=['COVID', 'Normal', 'Pneumonia'], normalize=False)
            plt.title(f'{phase.capitalize()} Confusion Matrix')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.savefig(f'{phase}_confusion_matrix.png')
            wandb.log({f'{phase}_confusion_matrix_image': wandb.Image(f'{phase}_confusion_matrix.png')})
            plt.close()

    return model

class HybridResNet18AlexNet(nn.Module):
    def __init__(self, num_classes=3):
        super(HybridResNet18AlexNet, self).__init__()

        # Load pre-trained AlexNet and ResNet18 without the final classification layers
        alexnet = models.alexnet(pretrained=False)
        resnet18 = models.resnet18(pretrained=False)

        # Modify AlexNet to accept single-channel images
        self.features_alexnet = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),  # Update the first conv layer's in_channels from 3 to 1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        # Add a convolutional layer to match the number of output channels from AlexNet to the input channels expected by ResNet18
        self.conv_match = nn.Conv2d(192, 64, kernel_size=1, stride=1, padding=0)

        # Use ResNet18 features from layer 1 to layer 4
        self.features_resnet18 = nn.Sequential(
            resnet18.layer1,
            resnet18.layer2,
            resnet18.layer3,
            resnet18.layer4
        )

        # Add a global average pooling layer and a fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 1 * 1, num_classes)

    def forward(self, x):
        # Pass the input through the AlexNet and ResNet18 features
        x = self.features_alexnet(x)
        x = self.conv_match(x)
        x = self.features_resnet18(x)

        # Apply global average pooling and flatten the output
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        # Pass through the fully connected layer
        x = self.fc(x)

        return x

# Initialize Weights and Biases
wandb.init(project='your-project-name', entity='your-team-name')

model = HybridResNet18AlexNet(num_classes=3)
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {total_params}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model_new = trained_model(model, criterion, optimizer, 10)
