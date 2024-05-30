import glob
import os
import shutil
import random

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

import timm
from tqdm import tqdm

# Size of training batch
BATCH_SIZE = 32
# Number of epochs to train (I don't have a CUDA gpu so this is set pretty low)
NUM_EPOCHS = 50
# Input dimensions for the model
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
# Total image data will be split 70% to training 30% to validation
TRAIN_SPLIT = 0.7

# Processes the dataset
class GenderDataset(Dataset):
    def __init__(self, data_dir, data_transform=None):
        self._images = ImageFolder(data_dir, transform=data_transform)

    def __len__(self):
        return len(self._images)

    def __getitem__(self, item):
        return self._images[item]

    @property
    def classes(self):
        return self._images.classes

# Basic image classification model setup
class GenericClassifierModel(torch.nn.Module):
    def __init__(self, num_classes):
        super(GenericClassifierModel, self).__init__()

        # Begin training with efficientnet, to bypass all the training of the model
        # for what an object is, how to distinguish foreground from background etc.
        self._template_model = timm.create_model('efficientnet_b1', pretrained=True)
        self._features = torch.nn.Sequential(*list(self._template_model.children())[:-1])

        self._classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(1280, num_classes)
        )

    def forward(self, input):
        """
        Forward propagate the network
        Args:
            input: Image

        Returns: Results

        """
        input = self._features(input)
        output = self._classifier(input)
        return output


def train(train_folder: str, val_folder: str):
    """
    Trains a model and exports it to an onnx file
    Args:
        train_folder: Path to the training folder
        val_folder: Path to the validation folder
    """
    image_transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToTensor()
    ])

    train_dataset = GenderDataset(data_dir=train_folder, data_transform=image_transform)
    val_dataset = GenderDataset(data_dir=val_folder, data_transform=image_transform)

    # Only shuffle for the training dataset, as that's the only one where the order matters
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Two classes, sorted alphabetically:
    # 0: Men
    # 1: Women
    model = GenericClassifierModel(num_classes=2)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    if torch.cuda.is_available():
        device_name = "cuda:0"
    else:
        device_name = "cpu"

    device = torch.device(device_name)
    model.to(device)

    for epoch in range(NUM_EPOCHS):
        # Train model
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc="Training..."):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # Validate model
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating..."):
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * labels.size(0)

        val_loss = running_loss / len(val_loader.dataset)

        progress = (epoch + 1) / NUM_EPOCHS
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} ({progress:0.1%}) - Loss: {train_loss}, Val Loss: {val_loss}")

    # Export model to onnx
    out_model_path = os.path.join(".", "models", "gender_classifier.onnx")
    dummy_input = torch.randn(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH)
    torch.onnx.export(model, dummy_input, out_model_path)

def sort_images(men_folder, women_folder, train_folder, val_folder):
    """
    Divides the total training sets into train and validation folders
    based on TRAIN_SPLIT
    Args:
        men_folder: Folder containing src images of men
        women_folder: Folder containing src images of women
        train_folder: Dst folder to copy the training images to
        val_folder: Dst folder to copy the validation images to
    """
    # Clear the folders
    for super_folder in [train_folder, val_folder]:
        for sub_folder in ["men", "women"]:
            shutil.rmtree(os.path.join(super_folder, sub_folder), ignore_errors=True)

    # Copy the data into the folders
    for gender_src_folder, gender_dst_folder in zip([men_folder, women_folder], ["men", "women"]):
        all_images = glob.glob(os.path.join(gender_src_folder, "*"))
        random.shuffle(all_images)
        train_len = int(len(all_images) * TRAIN_SPLIT)
        for i in range(train_len):
            train_path: str = os.path.join(train_folder, gender_dst_folder)
            os.makedirs(train_path, exist_ok=True)
            shutil.copy(all_images[i], train_path)
        for i in range(train_len, len(all_images)):
            val_path: str = os.path.join(val_folder, gender_dst_folder)
            os.makedirs(val_path, exist_ok=True)
            shutil.copy(all_images[i], val_path)


if __name__ == "__main__":
    men_folder = os.path.join(".", "data", "men")
    women_folder = os.path.join(".", "data", "women")

    train_folder = os.path.join(".", "data", "train")
    val_folder = os.path.join(".", "data", "validate")

    sort_images(men_folder, women_folder, train_folder, val_folder)
    train(train_folder, val_folder)
