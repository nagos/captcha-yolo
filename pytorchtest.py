#!/usr/bin/env python3

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from captcha_dataset import CaptchaDataset
from model import NeuralNetwork

alpha_conf = 1
alpha_class = 10
alpha_shift = 1


MODEL_FILE = "model.pt"

# dimensions
# 0 - picture
# 1 - feature
# 2 - none
# 3 - position
def custom_loss(output, target):
    # dimensions
    # 0 - picture
    # 1 - none
    # 2 - position
    # 3 - feature
    output_r = output.permute(0, 2, 3, 1)
    target_r = target.permute(0, 2, 3, 1)
    output_confidence = torch.sigmoid(output_r[:, 0, :, 0])
    target_confidence = target_r[:, 0, :, 0]

    target_mask = torch.where(target_confidence>0, 1, 0)
    output_category = nn.functional.softmax(output_r[:, 0, :, 1:11], -1)
    target_category = target_r[:, 0, :, 1:11]

    output_shift = torch.sigmoid(output_r[:, 0, :, 11])
    target_shift = target_r[:, 0, :, 11]
    

    loss_confidence = torch.mean((output_confidence - target_confidence)**2)
    loss_category = torch.mean(torch.sum((output_category-target_category)**2, -1)*target_mask)
    loss_shift = torch.mean((((output_shift-target_shift)**2)*target_mask))
    return alpha_conf*loss_confidence+alpha_class*loss_category+alpha_shift*loss_shift


batch_size = 64
training_data = CaptchaDataset(batch_size*200)
test_data = CaptchaDataset(batch_size*1)


train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model = NeuralNetwork().to(device)
print(model)

loss_fn = custom_loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

try:
    checkpoint = torch.load(MODEL_FILE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
except:
    pass

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")

epochs = 20
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, MODEL_FILE )
print("Done!")
