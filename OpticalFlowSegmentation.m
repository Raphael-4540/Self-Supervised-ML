function [ fillBW, mag ] = OpticalFlowSegmentation(I_pair,th,MinPixelsInConnectedRegion,SizeOfSmoothingDisk)
% Optical flow segmentation using two images.

opticFlow = opticalFlowFarneback('FilterSize',15,'NumPyramidLevels',3);
flow = estimateFlow(opticFlow,I_pair(:,:,2)); 

% Threshold for flow vector magnitude.
u = flow.Vx; 
v = flow.Vy;
mag = sqrt( u.^2 + v.^2 ); 
bin_idx = ( mag > th);

% Initial binary image from flow vector thresholding.
BW = bin_idx; 

% Remove regions with low connectivity, effectively a size filter.
BW2 = bwareaopen(BW,MinPixelsInConnectedRegion); 

% Close image:  Smooth mask to remove "over-articulation".
se = strel('disk',SizeOfSmoothingDisk);
closeBW = imclose(BW2,se); 

% Fill image.
fillBW = imfill(closeBW,'holes'); 












