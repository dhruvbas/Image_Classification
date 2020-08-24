import numpy as np

import os
from settings import APP_MODEL
import requests

# pytorch libraries
import torch
from torch import optim,nn
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
from torchvision import models,transforms


from PIL import Image
from urllib.request import urlopen
from settings import APP_MODEL

#Initialize the model
def init(device):
    path = os.path.join(APP_MODEL,'melanoma.pth')
    model = torch.load(path,map_location=device)
    return model

#Get the prediction
def prediction(model,pred,device):
    with torch.no_grad():
        for i, data in enumerate(pred):
            images = data
            #N = images.size(0)
            images = Variable(images).to(device)
            outputs = model(images)
            sm = torch.nn.Softmax()
            probabilities = sm(outputs)
            probs = probabilities.data.cpu().numpy()[0] 
            probdist = {}
            for j, prob in enumerate(probs):
                add_element(probdist,imageMapping(j), f'{float(prob):.10f}')
    return probdist

# Get disease type 

def imageMapping(num):
    switcher={
                0:'actinic_keratoses',
                1:'basal_cell_carcinoma',
                2:'benign_keratosis_like_lesions',
                3:'dermatofibroma',
                4:'melanocytic_nevi',
                5:'melanoma',
                6:'vascular_lesions'
             }
    return switcher.get(num,"Invalid")

# Add elements in a dictionary

def add_element(dict, key, value):
    if key not in dict:
        #dict[key] = []
        dict[key] = value
    #dict[key].append(value)

# Define a pytorch dataloader for this dataset
class HAM10000(Dataset):
    def __init__(self, url, transform=None):
        self.url = url
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, index):
        # Load data and get label
        X = Image.open(urlopen(self.url))
        #X = Image.open(self.url)
     

        if self.transform:
            X = self.transform(X)

        return X




   
