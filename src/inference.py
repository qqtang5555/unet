import torch
from src.model import get_unet_model
from albumentations import Compose, Normalize
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt

def predict(image_path, model, device):
    transform = Compose([
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image=image)['image'].unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(image)
        pred = torch.sigmoid(output).cpu().numpy()
        return (pred > 0.5).astype('uint8')[0, 0]

def visualize_prediction(image_path, pred):
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(cv2.imread(image_path)[..., ::-1])
    plt.subplot(1, 2, 2)
    plt.title('Predicted Mask')
    plt.imshow(pred, cmap='gray')
    plt.show()
