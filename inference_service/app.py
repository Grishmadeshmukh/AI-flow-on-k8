import os
from flask import Flask, render_template, request
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import torch.nn as nn

class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.layer1 = nn.Conv2d(1, 32, 3, 1)   
        self.layer2 = nn.Conv2d(32, 64, 3, 1) 
        self.drop_a = nn.Dropout(0.25)        
        self.drop_b = nn.Dropout(0.5)          
        self.dense1 = nn.Linear(9216, 128)    
        self.dense2 = nn.Linear(128, 10)       

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.drop_a(x)
        x = torch.flatten(x, 1)
        x = self.dense1(x)
        x = F.relu(x)
        x = self.drop_b(x)
        x = self.dense2(x)
        return F.log_softmax(x, dim=1)

# Flask app setup
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# load model
device = torch.device("cpu")
digit_model = DigitClassifier().to(device)
digit_model.load_state_dict(torch.load("/mnt/gd2574_model.pt", map_location=device))
digit_model.eval()

# transformations
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

@app.route('/')
def home_page():
    return render_template("mnist.html")

@app.route('/classify', methods=['POST'])
def classify_digit():
    if 'file' not in request.files:
        return "No file uploaded", 400
    file = request.files['file']
    if file.filename == '':
        return "Empty filename", 400

    image = Image.open(file).convert("L")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = digit_model(image)
        pred = output.argmax(dim=1, keepdim=True).item()

    return render_template("mnist.html", prediction=pred)

# run app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
