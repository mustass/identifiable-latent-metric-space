# pip install torch torchvision gdown
# download file from google drive:
#  https://drive.google.com/uc?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM
# and put it in the data/celeba folder

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

# Define the transformations: resize to 64x64 and center crop
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor()
])
# Download and load the CelebA dataset for both train and test splits
train_dataset = datasets.CelebA(root='/data', split='train', download=True, transform=transform)
test_dataset = datasets.CelebA(root='/data', split='test', download=True, transform=transform)

# Create DataLoaders without shuffling
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

# Initialize lists to store the images
images_ = []
for images, labels in train_dataloader:
    images_.append(images.numpy().transpose(0, 3, 2, 1))

np.save('/data/celeba/celeba_train_images.npy', np.concatenate(images_, axis=0))
print(f"Saved train images to celeba_train_images.npy")

images_ = []
for images, labels in test_dataloader:
    images_.append(images.numpy().transpose(0, 3, 2, 1))

np.save('/data/celeba/celeba_test_images.npy', np.concatenate(images_, axis=0))
print(f"Saved test images to celeba_test_images.npy")
