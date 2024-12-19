import torch
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from src.dataset import GrapeDataset, get_transform
from src.model import get_unet_model
from tqdm import tqdm

def train_model(image_dir, mask_dir, epochs=10, batch_size=4, lr=1e-4, device='cuda'):
    dataset = GrapeDataset(image_dir, mask_dir, transform=get_transform())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = get_unet_model().to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for images, masks in tqdm(dataloader):
            images, masks = images.to(device), masks.to(device).unsqueeze(1).float()
            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader)}")

    return model
