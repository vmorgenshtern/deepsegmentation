# Region Segmentation via Deep Learning and Convex Optimization

## This repository contains 
* an implementation of a classical algorithm for segmenting surfaces in point clouds, called Region Growing Segmentation (RGS). It is based on [RGS](http://www.pointclouds.org/documentation/tutorials/region_growing_segmentation.php)
* a deep learning pipeline for segmenting faces in 3D point clouds

## Getting familiar with the folder structure
* Root folder: 
  * Contains all available jupyter notebooks
  * Contains shared functions
  * Contains configuration files:
    * config.json: Base file for network configuration
    * generator_config.py: Dictionary to adjust the generated point clouds
* Classic_RGS: Contains all program code that is only linked with the RGS algorithm. Also input and output files related to RGS are stored here. Note: The point cloud generator is not configured to store its output in this folder. Point clouds need to be manually shifted to the input folder.
* data: Contains program code for the point cloud and patch datasets. Contains all data that is linked with the deep learning pipeline.
  * input_data: When a dataset is generated, data is automatically stored here
  * network_data: Configuration files and network configurations are stored here during training
  * output_data: Plots during training, testing results and prediction results are stored here.
  * temp_data: Temporary data is stored here and continuously overwritten. Used e.g. when running the voxelization notebook.
  * tensor_board_logs: Not used.
* Network: Contains program files linked to the deep learning pipeline: Preprocessing, Network, Postprocessing
* PolyhedronGenerator: Contains the program code for the polyhedron generator.
* utils: Contains utility program code

## Getting started
1. Region Growing Segmentation:
  * Open the Jupyter notebook "Classic_RegionGrowingSegmentation". 
  * Make sure that the input path is correctly set.
  * After running the notebook, find the files in the specified output path.
2. Deep Learning Pipeline:
  * You can evaluate existing models via the notebooks "Predict" or "Evaluate". The difference is, that Evaluate has access to the ground truth data and returns more information. 
  * You can train a new or existing model using the notebook "Training". 
     * By setting an experiment name, it is assured that results are separated from other configurations. 
     * To overwrite an existing experiment, uncomment the 'delete' line.
     * To continue training an existing experiment, uncomment the 'create_retrain_experiment' line. Make sure that the 'delete line' is commented.
  * Datasets will automatically be created in these notebooks, if not available. Make sure that "generator_config.py" is as intended.
  * All results are stored in data/output_data
  * After training is finished, you can call the notebook "Read_Training_Results" which will display results on the validation set and automatically generate an html version for archiving.
3. Independent notebooks:
  * Notebook "PolyhedronGenerator": You can use this to try the polyhedron generator. Data will be stored in data/temp_data
  * Notebook "Voxelization": You can observe how the patches in a point cloud look and how a patch looks compared to its voxelized version. Data will be stored in data/temp_data
  
## Requirements
* Python 3.6.8
* torch 1.1.0
* numpy 1.15.4
* numba 0.43.1
* plotly 3.10.0
* matplotlib 3.0.2
* sklearn 0.0
* scipy 1.1.0
* tensorboardx 1.7
* jupyter 1.0.0
