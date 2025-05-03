from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)


def get_dataloader(data_root: str, batch_size: int, train: bool = True):
    dataset = datasets.CIFAR10(
        root=data_root, train=train, download=True, transform=transform
    )
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train)
    return dataloader
