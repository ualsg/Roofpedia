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

def intersection(target_type, city_name, mask_dir):
    # predicted features
    print()
    print("Converting Prediction Masks to GeoJson Features")
    features = mask_to_feature(mask_dir)
    prediction = gp.GeoDataFrame.from_features(features, crs=4326) 

    # loading building polygons
    city = 'results/01City/' + city_name + '.geojson'
    city = gp.GeoDataFrame.from_file(city)[['geometry']]  
    city['area'] = city['geometry'].to_crs({'init': 'epsg:3395'}).map(lambda p: p.area)
    
    intersections= gp.sjoin(city, prediction, how="inner", op='intersects')
    intersections = intersections.drop_duplicates(subset=['geometry'])
    
    intersections.to_file('results/04Results/' + city_name + '_' + target_type + ".geojson", driver='GeoJSON')
    
    print()
    print("Process complete, footprints with " + target_type + " roofs are saved at results/04Results/" + city_name + '_' + target_type + ".geojson")
    return intersections


if __name__=="__main__":

    config = toml.load('config/predict-config.toml')
    city_name = config["city_name"]
    target_type = config["target_type"]
    mask_dir = os.path.join("results", "03Masks", target_type, city_name)
    
    intersection(target_type, city_name, mask_dir)
    
