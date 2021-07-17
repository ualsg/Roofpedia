import os
import sys
import collections
import toml
from tqdm import tqdm
import webp
import torch
from torch.nn import DataParallel
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, CenterCrop, Normalize

from src.losses import CrossEntropyLoss2d, mIoULoss2d, FocalLoss2d, LovaszLoss2d
from src.unet import UNet
from src.utils import plot
from src.train import get_dataset_loaders, train, validate

def loop():
    device = torch.device("cuda")

    if not torch.cuda.is_available():
        sys.exit("Error: CUDA requested but not available")

    # weighted values for loss functions
    # add a helper to return weights seamlessly
    try:
        weight = torch.Tensor([1.513212, 10.147043])
    except KeyError:
        if model["opt"]["loss"] in ("CrossEntropy", "mIoU", "Focal"):
            sys.exit("Error: The loss function used, need dataset weights values")

    # loading Model
    net = UNet(num_classes)
    net = DataParallel(net)
    net = net.to(device)

    # define optimizer 
    optimizer = Adam(net.parameters(), lr=lr)

    # select loss function, just set a default, or try to experiment
    if loss_func == "CrossEntropy":
        criterion = CrossEntropyLoss2d(weight=weight).to(device)
    elif loss_func == "mIoU":
        criterion = mIoULoss2d(weight=weight).to(device)
    elif loss_func == "Focal":
        criterion = FocalLoss2d(weight=weight).to(device)
    elif loss_func == "Lovasz":
        criterion = LovaszLoss2d().to(device)
    else:
        sys.exit("Error: Unknown Loss Function value !")

    #loading data
    train_loader, val_loader = get_dataset_loaders(target_size, batch_size, dataset_path)
    history = collections.defaultdict(list)

    # training loop
    for epoch in range(0, num_epochs):

        print("Epoch: " + str(epoch +1))
        train_hist = train(train_loader, num_classes, device, net, optimizer, criterion)
        
        val_hist = validate(val_loader, num_classes, device, net, criterion)
        
        print("Train loss: {:.4f}, mIoU: {:.3f}, {} IoU: {:.3f}, MCC: {:.3f}".format(
                train_hist["loss"], train_hist["miou"], target_type, train_hist["fg_iou"], train_hist["mcc"]))
        
        print("Validation loss: {:.4f}, mIoU: {:.3f}, {} IoU: {:.3f}, MCC: {:.3f}".format(
                 val_hist["loss"], val_hist["miou"], target_type, val_hist["fg_iou"], val_hist["mcc"]))
        
        for key, value in train_hist.items():
            history["train " + key].append(value)

        for key, value in val_hist.items():
            history["val " + key].append(value)

        if (epoch+1)%5 == 0:
            # plotter use history values, no need for log
            visual = "history-{:05d}-of-{:05d}.png".format(epoch + 1, num_epochs)
            plot(os.path.join(checkpoint_path, visual), history)
        
        if (epoch+1)%20 == 0:
            checkpoint = target_type + "-checkpoint-{:03d}-of-{:03d}.pth".format(epoch + 1, num_epochs)
            states = {"epoch": epoch + 1, "state_dict": net.state_dict(), "optimizer": optimizer.state_dict()}
            torch.save(states, os.path.join(checkpoint_path, checkpoint))

if __name__ == "__main__":
    config = toml.load('config/train-config.toml')

    num_classes = 2
    lr = config['lr']
    loss_func = config['loss_func']
    num_epochs = config['num_epochs']
    target_size = config['target_size']
    batch_size  = config['batch_size']

    dataset_path = config['dataset_path']
    checkpoint_path = config['checkpoint_path']
    target_type = config['target_type']
    # make dir for checkpoint
    os.makedirs(checkpoint_path, exist_ok=True)
    loop()