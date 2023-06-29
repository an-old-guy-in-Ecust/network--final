import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data.dataloader as dataloader


def load_dataset(batch_size):
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]))
    train_dataloader = dataloader.DataLoader(train_dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=8)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]))
    test_dataloader = dataloader.DataLoader(test_dataset, batch_size=batch_size,
                                            shuffle=False, num_workers=8)
    return train_dataloader, test_dataloader
