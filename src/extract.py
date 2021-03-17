import os
import toml

from tqdm import tqdm
from PIL import Image

import geopandas as gp
from geopandas.tools import sjoin
import numpy as np
import pandas as pd
from shapely.geometry import Point

from robosat.tiles import tiles_from_slippy_map
from robosat.features.parking import ParkingHandler


def mask_to_feature(mask_dir):

    handler = ParkingHandler()
    
    tiles = list(tiles_from_slippy_map(mask_dir))

    for tile, path in tqdm(tiles, ascii=True, unit="mask"):
        image = np.array(Image.open(path).convert("P"), dtype=np.uint8)
        mask = (image == 1).astype(np.uint8)
        handler.apply(tile, mask)

    # output feature collection
    feature = handler.jsonify()

    return feature

def intersection(rtype, area, mask_dir):
    # predicted features
    features = mask_to_feature(mask_dir)
    prediction = gp.GeoDataFrame.from_features(features) 

    # loading building polygons
    city = 'results/01City/' + area + '.geojson'
    city = gp.GeoDataFrame.from_file(city)[['geometry']]  
    city['area'] = city['geometry'].to_crs({'init': 'epsg:3395'}).map(lambda p: p.area)
    
    intersections= gp.sjoin(city, prediction, how="inner", op='intersects')
    intersections = intersections.drop_duplicates(subset=['geometry'])
    
    intersections.to_file('results/04Results/' + area + '_' + rtype + ".geojson", driver='GeoJSON')
    
    return intersections


if __name__=="__main__":

    config = toml.load('config/predict-config.toml')
    mask_dir = config["mask_dir"] 
    target_type = config["target_type"] 
    feature_output_path = os.path.join("results", str(target_type))
    mask_dir = 'results/03Predicted_Masks\Solar\Melbourne'

    intersection('Solar', 'Melbourne', mask_dir)