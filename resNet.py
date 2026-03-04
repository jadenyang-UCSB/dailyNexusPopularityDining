#This is for re-identification since BotSort for YOLO is pretty bad
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
from scipy.spatial.distance import cosine
from PIL import Image

base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

resnet_encoder = nn.Sequential(*(list(base_model.children()))[:-1])
resnet_encoder.eval()

gate = transforms.Compose([
    transforms.Resize((255,255)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

def colorVector(color):
    PILIMAGE = Image.fromarray(color)
    prepped = gate(PILIMAGE).unsqueeze(0)
    with torch.no_grad():
        vector = resnet_encoder(prepped)
    
    return vector.flatten().numpy()