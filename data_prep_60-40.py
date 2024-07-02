import os
import glob
import torch
import shutil
import itertools
import torch.nn as nn
import torch.optim as optim
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from pathlib import Path
from torch.nn import functional as F
from torchvision import datasets, models, transforms


covid_path = 'D:/covid19/dataset/COVID/images'
normal_path = 'D:/covid19/dataset/Normal/images'
pneumonia_path = 'D:/covid19/dataset/Viral_Pneumonia/images'


os.mkdir('./train')
os.mkdir('./test')

os.mkdir('./train/covid')
os.mkdir('./test/covid')

os.mkdir('./train/normal')
os.mkdir('./test/normal')

os.mkdir('./train/pneumonia')
os.mkdir('./test/pneumonia')

covid_train_len = int(np.floor(len(os.listdir(covid_path))*0.6))
covid_len = len(os.listdir(covid_path))

normal_train_len = int(np.floor(len(os.listdir(normal_path))*0.6))
normal_len = len(os.listdir(normal_path))

pneumonia_train_len = int(np.floor(len(os.listdir(pneumonia_path))*0.6))
pneumonia_len = len(os.listdir(pneumonia_path))


for trainimg in itertools.islice(glob.iglob(os.path.join(covid_path, '*.png')), covid_train_len):
    shutil.copy(trainimg, './train/covid')
    
for trainimg in itertools.islice(glob.iglob(os.path.join(normal_path, '*.png')), normal_train_len):
    shutil.copy(trainimg, './train/normal')
    
for trainimg in itertools.islice(glob.iglob(os.path.join(pneumonia_path, '*.png')), pneumonia_train_len):
    shutil.copy(trainimg, './train/pneumonia')


for testimg in itertools.islice(glob.iglob(os.path.join(covid_path, '*.png')), covid_train_len, covid_len):
    shutil.copy(testimg, './test/covid')

for testimg in itertools.islice(glob.iglob(os.path.join(normal_path, '*.png')), normal_train_len, normal_len):
    shutil.copy(testimg, './test/normal')

for testimg in itertools.islice(glob.iglob(os.path.join(pneumonia_path, '*.png')), pneumonia_train_len, pneumonia_len):
    shutil.copy(testimg, './test/pneumonia')
