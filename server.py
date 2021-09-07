from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import torchvision
import torchvision.transforms as transforms
import numpy as np
import json
import os
from CNNNet import CNNNet
import torch
import pickle
import matplotlib.pyplot as plt 
from PIL import Image


app = Flask(__name__) # Main object of the Flask application

model = CNNNet()
model.load_state_dict(torch.load("modelCnn.pth", map_location="cpu"))
model.eval()

@app.route ('/home', methods=["GET", "POST"]) # Function handler for /
def hello():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join('uploads', filename))
 
	
        return redirect(url_for("prediction", filename=filename))
    return render_template('home.html')

@app.route('/prediction/<filename>')
def prediction(filename):
    transforms = torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    pil_img = Image.open("uploads/" + filename)
    img = transforms(pil_img)
    probabilities = model(img.unsqueeze(0))

    softmax = torch.nn.functional.softmax(probabilities, dim=1)
    number_to_class = ['Crazing', 'Inclusion', 'Patches', 'Pitted', 'Rolled', 'Scratches']
    index = torch.argsort(softmax, dim=1)[0]
    probabilities = softmax.tolist()[0]
 
    predictions = {
                  "class1":number_to_class[index[5]],
                  "class2":number_to_class[index[4]],
                  "class3":number_to_class[index[3]],
                  "prob1":probabilities[index[5]],
                  "prob2":probabilities[index[4]],
                  "prob3":probabilities[index[3]],
              }
    return render_template('prediction.html', predictions=predictions)
    
if __name__ == '__main__':
    app.run("0.0.0.0", 5000) # Run the server on port 5000