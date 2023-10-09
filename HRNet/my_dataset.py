from PIL import Image
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, image_paths, image_labels, transforms=None):
        self.image_paths = image_paths
        self.image_labels = image_labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = Image.open(self.image_paths[item]).convert('RGB')
        label = self.image_labels[item]

        if self.transforms:
            image = self.transforms(image)
        return image, label

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels