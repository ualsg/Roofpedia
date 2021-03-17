import os
import sys
import argparse
import collections
from contextlib import contextmanager
import toml
from tqdm import tqdm
from PIL import Image
import torch
from torch.nn import DataParallel
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, CenterCrop, Normalize
from src.transforms import (
    JointCompose,
    JointTransform,
    JointRandomHorizontalFlip,
    JointRandomRotation,
    ConvertImageMode,
    ImageToTensor,
    MaskToTensor,
)
from src.datasets import SlippyMapTilesConcatenation
from src.metrics import Metrics
from src.losses import CrossEntropyLoss2d, mIoULoss2d, FocalLoss2d, LovaszLoss2d
from src.unet import UNet
from src.utils import plot


def get_dataset_loaders(target_size, batch_size, dataset_path):
    target_size = (target_size, target_size)
    path = dataset_path
    
    # using imagenet mean and std for Normalization
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    transform = JointCompose(
        [   
            JointTransform(ConvertImageMode("RGB"), ConvertImageMode("P")),
            JointTransform(Resize(target_size, Image.BILINEAR), Resize(target_size, Image.NEAREST)),
            JointTransform(CenterCrop(target_size), CenterCrop(target_size)),
            JointRandomHorizontalFlip(0.5),
            JointRandomRotation(0.5, 90),
            JointRandomRotation(0.5, 90),
            JointRandomRotation(0.5, 90),
            JointTransform(ImageToTensor(), MaskToTensor()),
            JointTransform(Normalize(mean=mean, std=std), None),
        ]
    )

    train_dataset = SlippyMapTilesConcatenation(
        [os.path.join(path, "training", "images")], os.path.join(path, "training", "labels"), transform
    )

    val_dataset = SlippyMapTilesConcatenation(
        [os.path.join(path, "validation", "images")], os.path.join(path, "validation", "labels"), transform
    )

    assert len(train_dataset) > 0, "at least one tile in training dataset"
    assert len(val_dataset) > 0, "at least one tile in validation dataset"

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, val_loader


def train(loader, num_classes, device, net, optimizer, criterion):
    num_samples = 0
    running_loss = 0

    # always two classes in our case
    metrics = Metrics(range(num_classes))
    # initialized model
    net.train()
    
    # training loop
    for images, masks, tiles in tqdm(loader, desc="Train", unit="batch", ascii=True):
        images = images.to(device)
        masks = masks.to(device)

        assert images.size()[2:] == masks.size()[1:], "resolutions for images and masks are in sync"

        num_samples += int(images.size(0))
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        for mask, output in zip(masks, outputs):
            prediction = output.detach()
            metrics.add(mask, prediction)

    return {
        "loss": running_loss / num_samples,
        "miou": metrics.get_miou(),
        "fg_iou": metrics.get_fg_iou(),
        "mcc": metrics.get_mcc(),
    }

def validate(loader, num_classes, device, net, criterion):
    num_samples = 0
    running_loss = 0

    metrics = Metrics(range(num_classes))

    with torch.no_grad():
        net.eval()

        for images, masks, tiles in tqdm(loader, desc="Validate", unit="batch", ascii=True):
            images = images.to(device)
            masks = masks.to(device)

            assert images.size()[2:] == masks.size()[1:], "resolutions for images and masks are in sync"

            num_samples += int(images.size(0))
            outputs = net(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item()

            for mask, output in zip(masks, outputs):
                metrics.add(mask, output)

        return {
            "loss": running_loss / num_samples,
            "miou": metrics.get_miou(),
            "fg_iou": metrics.get_fg_iou(),
            "mcc": metrics.get_mcc(),
        }

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

    # add in resume training if possible

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
    
    # log = Log(os.path.join(checkpoint_path, "log"), "log")
    # log.log("--- Hyper Parameters on Dataset: {} ---".format(dataset["common"]["dataset"]))
    # log.log("Batch Size:\t {}".format(model["common"]["batch_size"]))
    # log.log("Image Size:\t {}".format(model["common"]["image_size"]))
    # log.log("Learning Rate:\t {}".format(model["opt"]["lr"]))
    # log.log("Loss function:\t {}".format(model["opt"]["loss"]))
    # if "weight" in locals():
    #     log.log("Weights :\t {}".format(dataset["weights"]["values"]))
    # log.log("---")

    history = collections.defaultdict(list)

    # training loop
    for epoch in range(0, num_epochs):
        # log.log("Epoch: {}/{}".format(epoch + 1, num_epochs))
        print("Epoch: " + str(epoch +1))
        train_hist = train(train_loader, num_classes, device, net, optimizer, criterion)
        
        # log.log("Train loss: {:.4f}, mIoU: {:.3f}, {} IoU: {:.3f}, MCC: {:.3f}".format(
        #         train_hist["loss"], train_hist["miou"], target_type, train_hist["fg_iou"], train_hist["mcc"]))

        for key, value in train_hist.items():
            history["train " + key].append(value)

        # validate for each epoch
        val_hist = validate(val_loader, num_classes, device, net, criterion)

        # log.log("Validation loss: {:.4f}, mIoU: {:.3f}, {} IoU: {:.3f}, MCC: {:.3f}".format(
        #         val_hist["loss"], val_hist["miou"], target_type, val_hist["fg_iou"], val_hist["mcc"]))

        for key, value in val_hist.items():
            history["val " + key].append(value)

        if (epoch+1)%5 == 0:
            # plotter use history values, no need for log
            visual = "history-{:05d}-of-{:05d}.png".format(epoch + 1, num_epochs)
            plot(os.path.join(checkpoint_path, visual), history)
        
        if (epoch+1)%10 == 0:
            checkpoint = "checkpoint-{:05d}-of-{:05d}.pth".format(epoch + 1, num_epochs)
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