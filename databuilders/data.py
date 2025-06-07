from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import argparse
import matplotlib.pyplot as plt

from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE

def process_celeba_data(root):
    # Define the transformations: resize to 64x64 and center crop
    transform = transforms.Compose(
        [
            transforms.CenterCrop(148),
            transforms.Resize(64),
            transforms.ToTensor(),
            # transforms.Normalize(
            #     mean=[0.5 * 255.0, 0.5 * 255.0, 0.5 * 255.0],
            #     std=[0.5 * 255.0, 0.5 * 255.0, 0.5 * 255.0],
            # ),
        ]
    )

    # Download and load the CelebA dataset for both train and test splits
    train_dataset = datasets.CelebA(
        root=root, split="train", download=True, transform=transform
    )
    val_dataset = datasets.CelebA(
        root=root, split="valid", download=True, transform=transform
    )
    test_dataset = datasets.CelebA(
        root=root, split="test", download=True, transform=transform
    )

    # Create DataLoaders without shuffling
    train_dataloader = DataLoader(
        train_dataset, batch_size=128, shuffle=False, num_workers=4
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=128, shuffle=False, num_workers=4
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=4
    )

    # Initialize lists to store the images
    images_train = []
    labels_train = []
    for images, labels in train_dataloader:
        images_train.append(images.numpy().transpose(0, 2, 3, 1))
        labels_train.append(labels.numpy())

    np.save(
        f"{root}/celeba/celeba_train_images.npy", np.concatenate(images_train, axis=0)
    )
    np.save(
        f"{root}/celeba/celeba_train_labels.npy", np.concatenate(labels_train, axis=0)
    )
    print(f"Saved train images and labels to celeba_train_images.npy")

    images_val = []
    labels_val = []
    for images, labels in val_dataloader:
        images_val.append(images.numpy().transpose(0, 2, 3, 1))
        labels_val.append(labels.numpy())

    np.save(f"{root}/celeba/celeba_val_images.npy", np.concatenate(images_val, axis=0))
    np.save(f"{root}/celeba/celeba_val_labels.npy", np.concatenate(labels_val, axis=0))

    images_test = []
    labels_test = []
    for images, labels in test_dataloader:
        images_test.append(images.numpy().transpose(0, 2, 3, 1))
        labels_test.append(labels.numpy())

    np.save(
        f"{root}/celeba/celeba_test_images.npy", np.concatenate(images_test, axis=0)
    )
    np.save(
        f"{root}/celeba/celeba_test_labels.npy", np.concatenate(labels_test, axis=0)
    )


def process_fashion_mnist_data(root):

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    # Download and load the CelebA dataset for both train and test splits
    train_dataset = datasets.FashionMNIST(
        root=root, train=True, download=True, transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        root=root, train=False, download=True, transform=transform
    )

    # Create DataLoaders without shuffling
    train_dataloader = DataLoader(
        train_dataset, batch_size=128, shuffle=False, num_workers=4
    )
    val_dataloader = DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=4
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=4
    )

    # Initialize lists to store the images
    images_train = []
    labels_train = []
    for images, labels in train_dataloader:
        images_train.append(images.numpy().transpose(0, 2, 3, 1))
        labels_train.append(labels.numpy())

    np.save(f"{root}/train_images.npy", np.concatenate(images_train, axis=0))
    np.save(f"{root}/labels.npy", np.concatenate(labels_train, axis=0))
    print(f"Saved train images and labels to fashion_mnist/train_images.npy")

    images_val = []
    labels_val = []
    for images, labels in val_dataloader:
        images_val.append(images.numpy().transpose(0, 2, 3, 1))
        labels_val.append(labels.numpy())

    np.save(f"{root}/val_images.npy", np.concatenate(images_val, axis=0))
    np.save(f"{root}/val_labels.npy", np.concatenate(labels_val, axis=0))

    images_test = []
    labels_test = []
    for images, labels in test_dataloader:
        images_test.append(images.numpy().transpose(0, 2, 3, 1))
        labels_test.append(labels.numpy())

    np.save(f"{root}/test_images.npy", np.concatenate(images_test, axis=0))
    np.save(f"{root}/test_labels.npy", np.concatenate(labels_test, axis=0))


def process_data_svhn(root):

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    # Download and load the CelebA dataset for both train and test splits
    train_dataset = datasets.SVHN(
        root=root, split="train", download=True, transform=transform
    )
    test_dataset = datasets.SVHN(
        root=root, split="test", download=True, transform=transform
    )

    # Create DataLoaders without shuffling
    train_dataloader = DataLoader(
        train_dataset, batch_size=128, shuffle=False, num_workers=4
    )
    val_dataloader = DataLoader(
        train_dataset, batch_size=128, shuffle=False, num_workers=4
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=4
    )

    # Initialize lists to store the images
    images_train = []
    labels_train = []
    for images, labels in train_dataloader:
        images_train.append(images.numpy().transpose(0, 2, 3, 1))
        labels_train.append(labels.numpy())

    np.save(f"{root}/train_images.npy", np.concatenate(images_train, axis=0))
    np.save(f"{root}/labels.npy", np.concatenate(labels_train, axis=0))
    print(f"Saved train images and labels to svhn/train_images.npy")

    images_val = []
    labels_val = []
    for images, labels in val_dataloader:
        images_val.append(images.numpy().transpose(0, 2, 3, 1))
        labels_val.append(labels.numpy())

    np.save(f"{root}/val_images.npy", np.concatenate(images_val, axis=0))
    np.save(f"{root}/val_labels.npy", np.concatenate(labels_val, axis=0))

    images_test = []
    labels_test = []
    for images, labels in test_dataloader:
        images_test.append(images.numpy().transpose(0, 2, 3, 1))
        labels_test.append(labels.numpy())

    np.save(f"{root}/test_images.npy", np.concatenate(images_test, axis=0))
    np.save(f"{root}/test_labels.npy", np.concatenate(labels_test, axis=0))


def process_data_cifar10(root):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    # Download and load the CelebA dataset for both train and test splits
    train_dataset = datasets.CIFAR10(
        root=root, train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root=root, train=False, download=True, transform=transform
    )
    # Create DataLoaders without shuffling
    train_dataloader = DataLoader(  
        train_dataset, batch_size=128, shuffle=True, num_workers=4
    )
    val_dataloader = DataLoader(    
        test_dataset, batch_size=128, shuffle=False, num_workers=4
    )           
    test_dataloader = DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=4
    )
    # Initialize lists to store the images
    images_train = []
    labels_train = []
    for images, labels in train_dataloader: 
        images_train.append(images.numpy().transpose(0, 2, 3, 1))
        labels_train.append(labels.numpy())
    np.save(f"{root}/train_images.npy", np.concatenate(images_train, axis=0))
    np.save(f"{root}/labels.npy", np.concatenate(labels_train, axis=0))
    print(f"Saved train images and labels to cifar10/train_images.npy")
    images_val = []
    labels_val = []
    for images, labels in val_dataloader:
        images_val.append(images.numpy().transpose(0, 2, 3, 1))
        labels_val.append(labels.numpy())       
    np.save(f"{root}/val_images.npy", np.concatenate(images_val, axis=0))
    np.save(f"{root}/val_labels.npy", np.concatenate(labels_val, axis=0))
    images_test = []
    labels_test = []
    for images, labels in test_dataloader:
        images_test.append(images.numpy().transpose(0, 2, 3, 1))
        labels_test.append(labels.numpy())
    np.save(f"{root}/test_images.npy", np.concatenate(images_test, axis=0))
    np.save(f"{root}/test_labels.npy", np.concatenate(labels_test, axis=0))
    print(f"Saved test images and labels to cifar10/test_images.npy")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process CelebA dataset")
    parser.add_argument(
        "--dataset",
        choices=["celeba", "fashion_mnist", "tasic", "cifar"],
        default="celeba",
        type=str,
    )
    parser.add_argument("--root", type=str, default="")
    args = parser.parse_args()

    datasets_avail = {
        "celeba": process_celeba_data,
        "fashion_mnist": process_fashion_mnist_data,
        "svhn": process_data_svhn,
        "cifar":process_data_cifar10
    }
    datasets_avail[args.dataset](args.root)
