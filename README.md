# Self-Supervised-ML
Self supervised machine learning code for segmenting live cell imagery (Matlab)

This Matlab code is designed to be used with time-resolved live cell microscopy images (tiffs) for the automated segmentation of cells from background. 

The principle of self-supervised machine learning is that you simply load your images and hit Run - no parameter tuning needed, no training imagery required.  

Run from start to finish, the code uses consecutive pairs of images to generate training data of 'cells' and 'background' via dynamic feature vectors 
based on optical flow (unsupervised). These self-labeled pixels are then used to generate static feature vectors (entropy, gradient), 
which in turn are used to train a classifier model. The training data is updated every image in order to automatically adapt to temporal changes in cell morphologies
or background illumination.

The code was tested for high fidelity segmentation using five different modes of light microscopy: 
transmitted light, DIC, phase contrast, fluorescence and interference reflection microscopy.  

Six different cell lines were imaged to cover a range of morphologies and phenotypic dynamics using three cameras of differing resolutions.  

These test data sets can be downloaded here: 
https://zenodo.org/record/5167318#.YQ_R2ohKhPY

The associated manuscript for this work can be found here (although the latest version is under peer review as of this writing):
https://www.biorxiv.org/content/10.1101/2021.01.07.425773v1 

This code was tested on Matlab v2020a and v2021a using commercially available laptop computers running the Windows 10 operating system.

The segmented output files can be used for higher level applications such as cell tracking and declumping.
Also included in this package is a higher level declumping application.  
