from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import argparse

def process_celeba_data(root):
    # Define the transformations: resize to 64x64 and center crop
    transform = transforms.Compose([
        transforms.CenterCrop(64),
        transforms.Resize(64),
        transforms.ToTensor()
    ])
    
    # Download and load the CelebA dataset for both train and test splits
    train_dataset = datasets.CelebA(root=root, split='train', download=True, transform=transform)
    val_dataset = datasets.CelebA(root=root, split='valid', download=True, transform=transform)
    test_dataset = datasets.CelebA(root=root, split='test', download=True, transform=transform)

    # Create DataLoaders without shuffling
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

    # Initialize lists to store the images
    images_train = []
    labels_train = []
    for images, labels in train_dataloader:
        print(images.shape)
        print(labels.shape)

        images_train.append(images.numpy().transpose(0, 3, 2, 1))
        labels_train.append(labels.numpy())

    np.save(f'{root}/celeba/celeba_train_images.npy', np.concatenate(images_train, axis=0))
    np.save(f'{root}/celeba/celeba_train_labels.npy', np.concatenate(labels_train, axis=0))
    print(f"Saved train images and labels to celeba_train_images.npy")

    images_val = []
    labels_val = []
    for images, labels in val_dataloader:
        images_val.append(images.numpy().transpose(0, 3, 2, 1))
        labels_val.append(labels.numpy())

    np.save(f'{root}/celeba/celeba_val_images.npy', np.concatenate(images_val, axis=0))
    np.save(f'{root}/celeba/celeba_val_labels.npy', np.concatenate(labels_val, axis=0))

    images_test = []
    labels_test = []
    for images, labels in test_dataloader:
        images_test.append(images.numpy().transpose(0, 3, 2, 1))
        labels_test.append(labels.numpy())

    np.save(f'{root}/celeba/celeba_test_images.npy', np.concatenate(images_test, axis=0))
    np.save(f'{root}/celeba/celeba_test_labels.npy', np.concatenate(labels_test, axis=0))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process CelebA dataset')
    parser.add_argument('--root', type=str, default='/home/stasy/ssdb/celeba_manual')
    args = parser.parse_args()

    process_celeba_data(args.root)
