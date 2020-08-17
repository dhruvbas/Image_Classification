from flask import Flask, render_template, request
import requests
from flask import jsonify

# python libraties
import os
import numpy as np
import pandas as pd


# pytorch libraries
import torch
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms



#from urllib.request import urlopen
from predict import imageMapping, init, prediction, HAM10000



 # init flask app
app = Flask(__name__)


global model, graph



norm_mean = [0.76303834, 0.5456503, 0.5700455]
norm_std = [0.14092807, 0.15261318, 0.1699708]
input_size = 224

# Image Transform
train_transform = transforms.Compose([transforms.Resize((input_size,input_size)),transforms.RandomHorizontalFlip(),
                                      transforms.RandomVerticalFlip(),transforms.RandomRotation(20),
                                      transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
                                        transforms.ToTensor(), transforms.Normalize(norm_mean, norm_std)])

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = init(device)

@app.route('/')

def index():
    try:

        return 'hello'
    except Exception as e:
        return e

@app.route('/predict', methods = ['GET','POST'])

def predict():
    try:
        model.eval()
        content = request.get_json()
        clientID = content['ClientID']
        counter = 1
        imagedic = {"ClientID": clientID}
        while counter < len (content):
            
            name = "Image" + str(counter)
            url = content[name]
            img = HAM10000(url, transform=train_transform)
            pred_loader = DataLoader(img, batch_size=1, shuffle=False, num_workers=0)
            imagedic[name] = prediction(model,pred_loader,device)
            counter = counter+1
        return jsonify(imagedic)
    except Exception as e:
        return e

 
if __name__ == "__main__":
    #port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', debug=True)

