import torch
from src.model import get_unet_model
from src.inference import predict, visualize_prediction

if __name__ == "__main__":
    model = get_unet_model()
    model.load_state_dict(torch.load("unet_model.pth"))
    model = model.to("cuda")

    pred = predict("data/test_images/example.jpg", model, device="cuda")
    visualize_prediction("data/test_images/example.jpg", pred)

