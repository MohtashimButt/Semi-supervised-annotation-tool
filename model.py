import os
import cv2
import torchvision.transforms as transforms
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os
from torch import nn
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_inference(input_dir):
    output_dir = "Labels"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    # Load pre-trained DeepLabv3+ model
    model = models.segmentation.deeplabv3_resnet50(weights=models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT)

    # Modify last layer for binary segmentation
    num_classes = 1
    model.classifier[4] = nn.Sequential(
        nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1)),
        nn.Sigmoid())

    # Load model on CPU
    model.load_state_dict(torch.load("deeplabv3_small_ubaid.pth", map_location=torch.device('cpu')))


    for filename in os.listdir(input_dir):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img_path = os.path.join(input_dir, filename)
            image_pil = Image.open(img_path)
            image_tensor = transform(image_pil)
            image_tensor = image_tensor.unsqueeze(0)

            model.eval()
            with torch.no_grad():
                output = model(image_tensor)['out'] >= 0.5
                pred_mask = output.squeeze().cpu().numpy()

            # Save the predicted mask image with the same name as the input image
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, pred_mask * 255)
            print(f"Masks for {filename} saved successfully.")

    print("All images processed and masks saved successfully.")