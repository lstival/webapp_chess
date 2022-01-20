#Load model
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import io

import torch
import torchvision.transforms as transforms
from torchvision.io import read_image
from torchvision import models

#Flask options
import os
import numpy as np
import flask
#from flask import Session
from flask import flash, request, redirect, url_for
from werkzeug.utils import secure_filename

img_size = 224
device = 'cpu'
list_classes = ['Bishop','King','Knight','Pawn','Queen','Rook']
num_classes = len(list_classes)


def GetModel():
    model = models.efficientnet_b0(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))

    model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(1280, 512),nn.ReLU(inplace=True),
        nn.Linear(512, 256),nn.ReLU(inplace=True),
        nn.Linear(256, 128),nn.ReLU(inplace=True),
        nn.Linear(128, 64),nn.ReLU(inplace=True),
        nn.Linear(64, num_classes))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    return(model, optimizer)

transform = transforms.Compose(
    [
        # transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
        ])

def load_model():
    PATH = "model/chess_model.pth"
    model, optimizer = GetModel()
    model.load_state_dict(torch.load(PATH,  map_location=torch.device('cpu')))
    model.eval()
    return model

#COnfig path to upload files and allowed extensions
UPLOAD_FOLDER = "images/"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = flask.Flask(__name__, template_folder='templates')
# Check Configuration section for more details
SESSION_TYPE = 'redis'
SESSION_PERMANENT = False
SECRET_KEY = b'_5#y2L"F4Q8z\n\xec]/12'
app.config.from_object(__name__)
#Session(app)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(image, model):
    try:
        img = transform(image).unsqueeze(0)
        prediction = model(img)
        predicted_class = np.argmax(prediction[0].detach().numpy())
        #print(f"Predict Class: {list_classes[predicted_class]}")
        return list_classes[predicted_class]
    except:
        return "Invalid Image, please try another!"


@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html', result=None))

    if request.method == 'POST':
        # check if the post request has the file part

        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            #filepath = (os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #file.save(filepath)
            #image = read_image(filepath)
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes))
            model = load_model()
            model.to(device)
            prediction = predict_image(image, model)
            #os.remove(filepath)
            return(flask.render_template('main.html', result=prediction))
            
            #return redirect(url_for('download_file', name=filename))
            
    return(flask.render_template('main.html', result=None))

if __name__ == '__main__':
    app.run()