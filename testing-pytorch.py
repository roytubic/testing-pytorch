import torch
from torchvision.models import resnet50, ResNet50_Weights
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from PIL import Image
import numpy as np

# Load a pre-trained model with the correct weights parameter
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

# Load and preprocess an image
img = Image.open('./images (1).jpeg').convert('RGB')
img = preprocess_image(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Convert img to float32 and normalize to [0, 1]
img_float32 = np.array(Image.open('./images (1).jpeg').convert('RGB')).astype(np.float32) / 255

# Initialize GradCAM
target_layer = model.layer4[-1]
cam = GradCAM(model=model, target_layers=[target_layer])

# Generate CAM
grayscale_cam = cam(input_tensor=img)[0, :]

# Visualize the CAM
visualization = show_cam_on_image(img_float32, grayscale_cam, use_rgb=True)
Image.fromarray((visualization * 255).astype(np.uint8)).show()

