# %%
import os
import glob
import torch
import shutil
import itertools
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from pathlib import Path
from torch.nn import functional as F
from torchvision import datasets, models, transforms
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


# %%
print(torch.__version__)
print(torchvision.__version__)

# %%
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

# %%
data_images = {
    'train': datasets.ImageFolder('./train', data_transforms['train']),
    'validation': datasets.ImageFolder('./test', data_transforms['validation'])
}

# %%
dataloaders = {
    'train': torch.utils.data.DataLoader(data_images['train'], batch_size=32, shuffle=True, num_workers=0),
    'validation': torch.utils.data.DataLoader(data_images['validation'], batch_size=32,shuffle=True,num_workers=0)
}

# %%
len(data_images['validation'])



# %%
import torchvision.models as models



# %%
class HybridAlexDenseNet(nn.Module):
    def __init__(self, num_classes=3):
        super(HybridAlexDenseNet, self).__init__()
        self.num_classes = num_classes

        # Load pre-trained AlexNet
        self.alexnet = models.alexnet(pretrained=True)
        self.alexnet.features[0] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)
        
        # Load pre-trained DenseNet201
        self.densenet = models.densenet201(pretrained=True)
        self.densenet.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        
        # Modify the last layer of AlexNet for the desired number of output classes
        self.alexnet.classifier[-1] = nn.Linear(self.alexnet.classifier[-1].in_features, 256)
        
        # Modify the last layer of DenseNet for the desired number of output classes
        self.densenet.classifier = nn.Linear(self.densenet.classifier.in_features, 256)
        
        # Define the final linear layer to accept the concatenated feature representation
        self.final_linear = nn.Linear(256 + 256, num_classes)
    
    def forward(self, x):
        # Pass the input through AlexNet and DenseNet
        alex_out = self.alexnet(x)
        densenet_out = self.densenet(x)
        
        # Concatenate the output of AlexNet with the output of DenseNet
        combined_out = torch.cat((alex_out, densenet_out), dim=1)
        
        # Pass through the final linear layer to get the output
        return self.final_linear(combined_out)

# %%
# Create an instance of the corrected model and print its summary
model = HybridAlexDenseNet(num_classes=3)



# %%
model = model.to(device)

# %%
total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {total_params}")

# %%
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# %%
training_loss = []
validation_loss = []
training_accuracy = []
validation_accuracy = []

def trained_model(model, criterion, optimizer, epochs, scaler=None):
    best_accuracy = 0.0
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
            
            # Stop training if testing accuracy goes above 95%
            if phase == 'validation' and epoch_accuracy > 0.94:
                print("Validation accuracy above 95%! Training stopped.")
                return model

    return model

# %%
model_new = trained_model(model, criterion, optimizer, 10)

# %%
from sklearn.metrics import confusion_matrix, classification_report, f1_score

# Set model to evaluation mode
model_new.eval()

# Lists to store true labels and predicted labels
true_labels = []
predicted_labels = []

# Iterate over validation dataset
for inputs, labels in dataloaders['validation']:
    inputs = inputs.to(device)
    labels = labels.to(device)

    # Forward pass
    outputs = model_new(inputs)
    _, preds = torch.max(outputs, 1)

    # Append true and predicted labels
    true_labels.extend(labels.cpu().numpy())
    predicted_labels.extend(preds.cpu().numpy())

# Compute confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Compute classification report
class_report = classification_report(true_labels, predicted_labels)

# Compute F1 score
f1 = f1_score(true_labels, predicted_labels, average='weighted')

print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)
print("\nWeighted F1 Score:", f1)



