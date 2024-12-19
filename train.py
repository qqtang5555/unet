from src.train import train_model

if __name__ == "__main__":
    model = train_model("data/images", "data/masks", epochs=10)
    torch.save(model.state_dict(), "unet_model.pth")
