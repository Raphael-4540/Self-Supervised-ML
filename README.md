# Self-Supervised-ML
Self-supervised machine learning code for segmenting live cell imagery (Matlab)

SSL_Demo_2.m : main program for self supervised machine learning segmentation

SSL_Declumping_2.m : main program for declumping application (applied to output of SSL_Demo_2.m)

For convenience and ease of use this code has been grouped with the accompanying Demo data for download from the following website:

https://zenodo.org/record/5193696#.YRZU9IhKhPY

However, if you prefer to work with the code in the GitHub environment and combine with data from Zenodo, please proceed as follows:

1. Download the demo data set from Zenodo: 
2. Combine the data folders with this GitHub code as shown in the "Code_Data_Directory_Structure_Image.JPG" image
3. Open SSL_Demo_2.m or SSL_Declumping_2.m in Matlab and hit Run. 

The principle of self-supervised machine learning is that you simply load your images and Run - no parameter tuning needed, no training imagery required.  

Run from start to finish, the code uses consecutive pairs of images to generate training data of 'cells' and 'background' via dynamic feature vectors 
based on optical flow (unsupervised). These self-labeled pixels are then used to generate static feature vectors (entropy, gradient), 
which in turn are used to train a classifier model. The training data is updated every image in order to automatically adapt to temporal changes in cell morphologies
or background illumination.

The code was tested for high fidelity segmentation using five different modes of light microscopy: 
transmitted light, DIC, phase contrast, fluorescence and interference reflection microscopy.  

Six different cell lines were imaged to cover a range of morphologies and phenotypic dynamics using three cameras of differing resolutions.  

The associated manuscript for this work can be found here (although the latest version is under peer review as of this writing):
https://www.biorxiv.org/content/10.1101/2021.01.07.425773v1 

This code was tested on Matlab v2020a and v2021a using commercially available laptop computers running the Windows 10 operating system.

