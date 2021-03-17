import os
import torch
import toml

from src.predict import predict
from src.extract import intersection

config = toml.load('config/predict-config.toml')
    
city_name = config["city_name"]
target_type = config["target_type"]
tiles_dir = os.path.join("results", '02Images', city_name)
mask_dir = os.path.join("results", "03Masks", target_type, city_name)
tile_size =  config["img_size"]

# load checkpoints
device = torch.device("cuda")
if target_type == "Solar":
    checkpoint_path = config["checkpoint_path"]
    checkpoint_name = config["solar_checkpoint"]
    chkpt = torch.load(os.path.join(checkpoint_path, checkpoint_name), map_location=device)

elif target_type == "Green":
    checkpoint_path = config["checkpoint_path"]
    checkpoint_name = config["green_checkpoint"]
    chkpt = torch.load(os.path.join(checkpoint_path, checkpoint_name), map_location=device)

predict(tiles_dir, mask_dir, tile_size, device, chkpt)

intersection(target_type, city_name, mask_dir)
