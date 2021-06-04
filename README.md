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
* A ready to use dataset of building footprints identified with Green Roofs and/or Solar roofs by Roofpedia.
* A straight forward pipeline to run prediction on your own satellite image dataset
* A guide on how you can tweak the pipeline to detect and tag roof features to OSM building footprints (coming up)

## Running Roofpedia 
Steps:
1. Install prequisites
1. Save a geojson file of building polygons (can be done in QGIS)
2. Extract the satellite tiles in slippymap format (done in QGIS)
3. Put the geojson and satellite tiles in their corresponding folders
4. run extract and predict
5. get result!
### 1. Prerequisites

You could use `environment.yml` to create a conda environment for Roofpedia

  ```sh
  conda env create -f environment.yml
  ```

For non-gpu users, delete `cudatoolkit=11.1` from environment.yml to run the inference on CPU.

### Data Preparation

For inference, datasets should be processed and placed in the `results` folder. 

A sample dataset is provided in the current results folder for illustration. 

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


Save a geojson file of building polygons (can be done in QGIS)
Extract the satellite tiles in slippymap format (can be done in QGIS)
### Prediction
Predictions can be carried out by running the following sample code. The name of the city depends on the name of each dataset.

 ```sh
  python predict_and_extract.py <city_name> <type>
  ```

so for example, when predicting Solar Panels on buildings in Berlin, use:

 ```sh
  python predict_and_extract.py Berlin Solar
  ```
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


