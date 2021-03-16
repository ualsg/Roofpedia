
import os
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize

from tqdm import tqdm
from PIL import Image
import toml

from robosat.datasets import BufferedSlippyMapDirectory
from robosat.unet import UNet
from robosat.config import load_config
from robosat.transforms import ConvertImageMode, ImageToTensor
from robosat.colors import make_palette

from robosat.tiles import tiles_from_slippy_map
from robosat.features.parking import ParkingHandler

def predict(tiles_dir, mask_dir, tile_size, device, chkpt):
    # load device
    net = UNet(2).to(device)
    net = nn.DataParallel(net)
    net.load_state_dict(chkpt["state_dict"])
    net.eval()

    # preprocess and load
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    transform = Compose([ConvertImageMode(mode="RGB"), ImageToTensor(), Normalize(mean=mean, std=std)])

    # tiles file, need to get it again, or do we really need it? why not just predict
    directory = BufferedSlippyMapDirectory(tiles_dir, transform=transform, size=tile_size)
    assert len(directory) > 0, "at least one tile in dataset"

    # loading data
    loader = DataLoader(directory, batch_size=1)

    # don't track tensors with autograd during prediction
    with torch.no_grad():
        for images, tiles in tqdm(loader, desc="Eval", unit="batch", ascii=True):
            images = images.to(device)
            outputs = net(images)

            # manually compute segmentation mask class probabilities per pixel
            probs = nn.functional.softmax(outputs, dim=1).data.cpu().numpy()

            for tile, prob in zip(tiles, probs):
                x, y, z = list(map(int, tile))

                prob = directory.unbuffer(prob)
                mask = np.argmax(prob, axis=0)
                mask = mask*200
                mask = mask.astype(np.uint8)

                palette = make_palette("dark", "light")
                out = Image.fromarray(mask, mode="P")
                out.putpalette(palette)

                os.makedirs(os.path.join(mask_dir, str(z), str(x)), exist_ok=True)
                path = os.path.join(mask_dir, str(z), str(x), str(y) + ".png")
                out.save(path, optimize=True)
    
    print("Prediction Done, saved masks to " + mask_dir)

def mask_to_feature(output_folder):

    handler = ParkingHandler()
    
    tiles = list(tiles_from_slippy_map(mask_dir))

    for tile, path in tqdm(tiles, ascii=True, unit="mask"):
        image = np.array(Image.open(path).convert("P"), dtype=np.uint8)
        mask = (image == 1).astype(np.uint8)
        handler.apply(tile, mask)

    # output feature collection
    handler.save(os.path.join(output_path, "feature.geojson"))

if __name__=="__main__":
    config = toml.load('predict config.toml')

    checkpoint_path = config["checkpoint_path"]
    checkpoint_name = config["checkpoint_name"]
    tile_size =  config["img_size"]
    tiles_dir = config["img_dir"] 
    mask_dir = config["mask_dir"] 
    device = torch.device("cuda")
    # load checkpoint
    chkpt = torch.load(os.path.join(checkpoint_path, checkpoint_name), map_location=device)
    
    predict(tiles_dir, mask_dir, tile_size, device, chkpt)