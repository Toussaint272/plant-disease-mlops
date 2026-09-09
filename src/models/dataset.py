"""Chargement des images et data augmentation."""
import yaml
from torchvision import datasets, transforms

params = yaml.safe_load(open("params.yaml", encoding="utf-8"))

IMG_SIZE = params["data"]["image_size"]
AUG = params["augmentation"]

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def train_transforms():
    return transforms.Compose([
        transforms.RandomResizedCrop(
            IMG_SIZE,
            scale=tuple(AUG["random_resized_crop_scale"])
        ),
        transforms.RandomHorizontalFlip(p=AUG["hflip"]),
        transforms.RandomVerticalFlip(p=AUG["vflip"]),
        transforms.RandomRotation(degrees=AUG["rotation"]),
        transforms.ColorJitter(
            brightness=AUG["color_jitter"],
            contrast=AUG["color_jitter"],
            saturation=AUG["color_jitter"],
            hue=0.05
        ),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=5)],
            p=AUG["gaussian_blur_p"]
        ),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def eval_transforms():
    return transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.14)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def load_dataset(split: str, train: bool = False):
    return datasets.ImageFolder(
        root=f"data/processed/{split}",
        transform=train_transforms() if train else eval_transforms()
    )
