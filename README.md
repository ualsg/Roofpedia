<p align="center">
  <a href="https://ual.sg/">
    <img src="images/logo.jpg" alt="Logo">
  </a>
  <h3 align="center">Roofpedia - Mapping Roofscapes with AI</h3>
  <p align="center">
    <br />
    <a href="https://ual.sg/project/roofpedia/"><strong>Explore Sustainable Roofscapes Around the World »</strong></a>
  </p>
</p>

This is the official repo of Roofpedia, an open registry of green roofs and solar roofs across the globe identified by Roofpedia through deep learning.

In this repo you will find:
* A ready to use dataset of 1,812 manually labelled polygons of rooftop greenery and solar panels covering 8 cities. 
* A ready to use dataset of building footprints identified with Green Roofs and/or Solar roofs by Roofpedia. Download data and labels [here](https://drive.google.com/file/d/13R5hOthwtm8IR-ke_IqFLkJBOTZ0Ys-s/view?usp=sharing)
* A straight forward pipeline to run prediction on your own satellite image dataset
* A guide on how you can tweak the pipeline to detect and tag roof features to OSM building footprints (coming up)

## Running Roofpedia 
Steps:
1. Install prequisites
2. Download and extract weights and sample dataset
3. run predict_and_extract.py 
4. get result!
### 1. Prerequisites

You could use `environment.yml` to create a conda environment for Roofpedia

  ```sh
  conda env create -f environment.yml
  ```

For non-gpu users, delete `cudatoolkit=11.1` from environment.yml to run the inference on CPU.

### 2. Data Preparation

Download the pretrained weights and sample dataset [here](https://drive.google.com/file/d/1uRsuXxSEhDEHaa8CoMmncpbClJ2fapJx/view?usp=sharing) and extract them to the root folder of the repo. 

For custom inference, datasets should be processed and placed in the `results` folder. See more details in later sections.


### Prediction

Predictions can be carried out by running the following sample code. The name of the city depends on the name of each dataset.

 ```sh
  python predict_and_extract.py <city_name> <type>
  ```

A sample dataset is provided in the results folder for prediction, just run

 ```sh
  python predict_and_extract.py Luxembourg Solar
  ```

for Greenroof, run

 ```sh
  python predict_and_extract.py Luxembourg Green
  ```

See the result in `04Result` folder

### Custom Dataset
Custom Dataset pairs can be created with QGIS using tiling functions. 
1. Create a WMTS satellite tile connection with any WMTS server. You can use Mapbox's WMTS server for good quality images.
2. With QuickOSM, query and download the building footprint of a desired area for prediction.
3. Save the building polygons to `01City` folder.
4. Callup QGIS toolbar (`Ctrl + Alt +T`), in `Raster Tools`, choose `Generate XYZ Tiles(Directory)` to generate satellite tiles for the area by using Canvas Extent. Use Zoom 19 and save to `02Images/Cityname`
5. You are now ready for prediction

A unified script in extracting building polygons and downloading satellite tiles from Mapbox is a work-in-progress.
### Custom Dataset File Structure
The structure of the `results` folder is as follows: 

📂results  
 ┣ 📂01City   
 ┃- ┗ 📊Cityname1.geojson  
 ┃- ┗ 📊Cityname2.geojson  
 ┣ 📂02Images  
 ┃--- ┗ 📂Cityname1  
 ┃--- ┗ 📂Cityname2  
 ┣ 📂03Masks  
 ┃--- ┗ 📂Green  
 ┃---   ┗ 📂Cityname1  
 ┃---   ┗ 📂Cityname2  
 ┃--- ┗ 📂Solar  
 ┃---   ┗ 📂Cityname1  
 ┃---   ┗ 📂Cityname2  
 ┣ 📂04Results  
 ┃- ┗ 📊Cityname1_Green.geojson  
 ┃- ┗ 📊Cityname1_Solar.geojson  
 ┃- ┗ 📊Cityname2_Green.geojson  
 ┃- ┗ 📊Cityname2_Solar.geojson  

`01City` contains geojson files of building polygons  
`02Images` contains a slippymap directory of satellite images. For the pre-trained models, a zoom level of 19 is required.  
`03Masks` contains predicted masks of each tile according to object type
`04Results` contains final cleaned building footprints tagged with the specific object type



### Training
By preparing your own labels, you can train your own model. Training options can be set under `config/train-config.toml`

 ```sh
  python train.py
  ```
<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- CONTACT -->

<!-- ## Contact

Your Name - [@your_twitter](https://twitter.com/your_username) - email@example.com

Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name) -->



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

Roofpedia is made possible by using the following packages

* [PyTorch](https://pytorch.org/)
* [GeoPandas](https://geopandas.org/)
* [Robosat](https://github.com/mapbox/robosat) - 
loading of slippy map tiles for training and mask to feature function is borrowed from robosat


