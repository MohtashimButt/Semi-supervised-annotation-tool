import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from torch import nn
from torchvision import datasets, models

weights = "deeplabv3_small.pth"
model = models.segmentation.deeplabv3_resnet50(weights = models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT)
num_classes = 1
model.classifier[4] = nn.Sequential(
    nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1)),
    nn.Sigmoid())
model = model.load_state_dict(torch.load(weights))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to apply the model and save the mask
def generate_mask(image_path, model):
    # Load the image
    image = Image.open(image_path).convert("RGB")
    # Apply transformation
    input_image = transform(image).unsqueeze(0)
    # Forward pass
    with torch.no_grad():
        output = model(input_image)['out'][0]
    # Get the predicted mask
    predicted_mask = output.argmax(0)
    predicted_mask = np.array(predicted_mask.cpu(), dtype=np.uint8)
    # Save the mask
    mask_path = os.path.join("Labels", os.path.basename(image_path))
    cv2.imwrite(mask_path, predicted_mask)


generate_mask("Unlabelled_Images\GN_006_IMG_1930_1.png", model)
