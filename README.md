# Deep Object Removal

Image completion is a challenging problem because it requires a high-level recognition of scenes. This project tries to achieve object removal from images and get the base image reconstructed based on surrounding colours and objects using conditional GANs.

# Overview
This project is an implementation of cGANs discussed in the paper for [\[General Image Completion\]](https://www.dropbox.com/s/e4l19y9ggqqk2yf/0360.pdf?dl=1)  
The models are tweaked a little and implemented to remove objects from images and reconstruct the image without the object.

# Example Usage
![](images/example/example.gif)

## Hot Keys
\[Esc\]: To quit the windowed application.  
\[f\]: To filter out the masked object.  
\[n\]: To go to the next image.  
\[r\]: To refresh and undo all the masking in the current image.  

# Description
## Files
### images/
The folder that contains the images to be used in the project. Currently the project requires images of dimensions 400 x 400 which can be changed in the main.py file. 

###  model/
This folder contains the pretrained model that is trained on mscoco dataset and the model definition file which is written in tensorflow.

### main.py
The main file to run the program. The code runs as an OpenCV windowed application.

### requirements.txt
The requirements file for the project

## Installation
To install the dependencies type

```
sudo pip3 install -r requirements.txt
```

## Run
To run the application type

```{python}
python3 main.py
```

This will run the demo as an OpenCV application 

## Dependencies
The project requires the following packages:  


OpenCV and OpenCV_python 3.3.0.10  
Tensorflow 1.10.1  
Numpy 1.13.3