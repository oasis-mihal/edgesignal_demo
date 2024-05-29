import os

import torchvision.transforms as transforms

from src.Trainer import GenderDataset


def test_dataset(mocker):
    image_transform = transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.ToTensor()
    ])
    test_data_dir = os.path.join(".", "tests", "test_data_dir")
    test_dataset = GenderDataset(data_dir=test_data_dir, data_transform=image_transform)

    assert len(test_dataset) == 5
    assert test_dataset.classes == ["class_1", "class_2"]
    assert test_dataset[0][0].shape[1] == 200