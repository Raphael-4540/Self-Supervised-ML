% SSL_Demo (August 2021)
%
% This code was tested on Matlab v2020a and v2021a using commercially available 
% laptop computers running the Windows 10 operating system.
%
% The principle of self-supervised ML is that you simply load your images 
% and hit Run - no parameter tuning needed, no training imagery required.
%
% For first time users, we recommend evaluating the code section by section
% (Ctrl+Enter) and reading the section comments which incorporate instructions 
% on loading your files and explanations of how the code works.
%
% Demo image pairs can be found here for testing purposes:
% https://zenodo.org/record/5167318#.YQ_R2ohKhPY
%
% The demo images files are not large so you can download onto your computer 
% and read into the Matlab workspace using the example 'dir_str' format below.
%
% Measured run times per frame on a laptop computer running Windows 10 for
% a variety of frame sizes:
% 512 x 672, 8 bit: 5 sec
% 1216 x 1920, 8 bit: 7 sec
% 2050 x 2050, 8 bit: 45 sec
%
% Overview
% Run from start to finish, this code uses consecutive pairs of images to 
% generate training data of 'cells' and 'background' via dynamic feature 
% vectors based on optical flow (unsupervised). These self-labeled pixels 
% are then used to generate static feature vectors (entropy, gradient), 
% which in turn are used to train a classifier model. The training data is 
% updated every image.
%
% The default is to ignore cells overlapping the border but this can be
% toggled by commenting out the imclearborder() line in Part 4. 
% 
% Input: tiff files from live cell microscopy experiment
% Output: 
%   (1) BW_cube_fv(): Binary images in which segemented cell pixels are
%       ones and segmented background pixels are zeroes.
%   (2) Video of segmented imagery (.avi)
%
% Functions called:
%   ReadDataCube(): used to read in tiff files into a 3D matrix
%   OpticalFlowSegmentation(): used to automate size filtering (Part 3)
%   SS_Training_Data(): uses entropy to determine a
%       reasonable optical flow threshold for labeling (Part 4). 
%   Bayes_S_trainClassifier(): Naive Bayes classifier for entropy feature
%       vector only.
%   Bayes_S_G_trainClassifier(): Naive Bayes classifier for entropy and
%       gradient feature vectors.
%
% Start fresh
clear; clc;
close all;

% USER INPUT REQUIRED!!
% Select data source. ONLY files to be analyzed of data type TIFF should be
% in this folder. e.g.: dir_str = 'C:\Users\Marc\Desktop\Fig3a_pair\Fig3a_pair'
dir_str = 'C:\Users\Marc\Desktop\Fig3a_pair\Fig3a_pair'

%% Part 1: Decide which static feature vectors to generate 

entr = true; % This should always be true. Entropy is currently a required feature vector
grad = true; % Gradient feature vector (recommended).
S_nhood = true(7); % entropyfilt() input

%% Part 2  Read in the imagery and microscope meta data.

I = ReadDataCube(dir_str); % Currently written for Windows OS!
[rows,cols,NumOfFrames] = size(I);

% Hard code NumOfFrames to sample images
%NumOfFrames = 2;

BW_OF_pair = false(rows,cols,2); % Used to automate minimum cell size value (Part 3)
BW_cube_fv = false(rows,cols,NumOfFrames-1); % Images of segmented cells (white) and background (black)
pre_BW_cube = false(rows,cols,1); % Before populating BW_cube_fv use this buffer to make sure no cells touch frame edges


%% Part 3: Use conservative optical flow threshold to automate minimal cell size filter

% Threshold for optical flow vector magnitude.
th = 0.04; % Conservative threshold for automating minimum cell size value

% Morphological processing parameters.
MinPixelsInConnectedRegion = 600; % Default cell size
SizeOfSmoothingDisk = 5; % Default smoothing disk (5). Works well on all test images so far MPR 7/20/2020

moving = I(:,:,2); % This is the frame being analyzed.
fixed = I(:,:,1);% This is the background frame for optical flow.
clear I_pair
% Perform alignment and trim border.
I_pair(:,:,1) = fixed;
I_pair(:,:,2) = moving;

% Compute optical flow segmentation (not self-supervised)
[BW_OF_pair(:,:,1), mag] = OpticalFlowSegmentation(I_pair,th,MinPixelsInConnectedRegion,SizeOfSmoothingDisk);

k =1;
%SegFig_OF = figure; 
stats_OF = regionprops('table',BW_OF_pair(:,:,k),'Area','Centroid');
Area_OF{k} = stats_OF.Area;

B_OF = bwboundaries(BW_OF_pair(:,:,k));
NF_OF = length(B_OF);
    
% Display. 
%imshow(imadjust(I(:,:,2))); colormap bone; hold on;
%for i = 1:length(B_OF)
%    bdd_OF = B_OF{i};
%    plot(bdd_OF(:,2), bdd_OF(:,1), 'r', 'LineWidth', 1)
%end
%centroid = stats_OF.Centroid;

%if ~isempty(centroid)
%    scatter(centroid(:,1),centroid(:,2),'r+')
%end
%axis equal; title(['OF Initialize: Use for Estimating Minimum Cell Size']);   
%pause(0.01); hold off;

if isnan(mean(Area_OF{1,1})) == 1 % In case no cells are segmented
    MinPixelsInConnectedRegion = 550;  
else
    MinPixelsInConnectedRegion = round(mean(Area_OF{1,k})./2,0); % Divide mean area by two to be conservative
end


%% Part 4: Label, train and segment using previously generated model

% Classification
label = {'background', 'cell'};

tic
extra_S = 4e3; % Camera dependent: 4e3 works well for sensors sizes of 2 Mpixel or more.
for k = 2:NumOfFrames % Select a frame.
    moving = I(:,:,k); % This is the frame being analyzed.
    fixed = I(:,:,k-1);% This is the background frame for optical flow.
    clear I_pair
    % Form image pair matrix
    I_pair(:,:,2) = moving;
    I_pair(:,:,1) = fixed;
    
    % Compute optical flow for background and cell labeling by
    % self-supervising with image entropy
    %formatSpec = 'Self Tuning Thresholds for Training Data Generation';
    %str = sprintf(formatSpec)
    [bg_train, cell_train] = SS_Training_Data(I_pair,MinPixelsInConnectedRegion, S_nhood, extra_S);
    
    % Create OF generated BW training masks for subsequent feature vector training data
    % generation
    BW_bg_train = logical(bg_train);
    BW_cell_train = logical(cell_train);
    
    % Form entropy feature vector table (static) for input into model
    if entr == true % Should al     
        % Form entropy image
        test_img_entropy = entropyfilt(I(:,:,k), S_nhood);
        bin_img_entropy = (test_img_entropy == 0); % protect against entropyfilt() = 0 elements so as not to confuse with mask zeros
        test_img_entropy = test_img_entropy + 0.001.*bin_img_entropy; 
        
        % Background gradient feature vector
        bg_train_S = double(BW_bg_train).*test_img_entropy; % entropy image of background
        bg_ind_S = find(bg_train_S);
        bg_fv_S = bg_train_S(bg_ind_S); % entropy feature vector of background
        bg_fv_class = cell(size(bg_fv_S)); % create cell for classifiers
        for m = 1:size(bg_fv_S) 
            bg_fv_class(m) = label(1,1);
        end
        % Background entropy feature vector 
        bg_fv_S_table = table(bg_fv_S, bg_fv_class, 'VariableNames', {'entropy', 'Label'});
        
        % Cell entropy feature vector
        cell_train_S = double(BW_cell_train).*test_img_entropy; % Entropy image of cells
        cell_ind_S = find(cell_train_S); % Filter out mask pixels with value zero
        cell_fv_S = cell_train_S(cell_ind_S);% entropy feature vector  
        cell_fv_class = cell(size(cell_fv_S)); % create cell for category
        for m = 1:size(cell_fv_S)
            cell_fv_class(m) = label(1,2);
        end
        
        bg_cell_fv_class = cat(1, bg_fv_class, cell_fv_class); %

        % Input table for classification model
        cell_fv_S_table = table(cell_fv_S, cell_fv_class, 'VariableNames', {'entropy', 'Label'}); % feature vector values and labels
       
        % Input table for classification model
        fv_table_S = cat(1, bg_fv_S_table, cell_fv_S_table);  
        bg_cell_fv_S = [bg_fv_S; cell_fv_S]; 
        
        % Entropy feature vector of image to be classified
        test_fv_S = test_img_entropy(1:(rows*cols))'; % feature vector of all pixels
        if grad == false
            test_fv_table = table(test_fv_S, 'VariableNames', {'entropy'}); 
        end      
        
    end
       
    %  Generate gradient feature vector (static) and append to entropy
    %  table
    if grad == true
        
        % Form gradient image
        [test_img_gradient, gdir] = imgradient(I(:,:,k), 'intermediate');
        bin_img_gradient = (test_img_gradient == 0); % find gradient = 0 elements so as not to confuse with mask zeros
        test_img_gradient = test_img_gradient + 0.001.*bin_img_gradient; % replace those zeros with small number (0.001)
        
        % Background gradient feature vector
        bg_train_G = double(BW_bg_train).*test_img_gradient; % Gradient image of background
        bg_ind_G = find(bg_train_G);
        bg_fv_G = bg_train_G(bg_ind_G); % Gradient feature vector for background 
        
        % Cell gradient feature vector
        cell_train_G = double(BW_cell_train).*test_img_gradient; % Associated gradient image of cells
        cell_ind_G = find(cell_train_G);
        cell_fv_G = cell_train_G(cell_ind_G);% Gradient feature vector for cells
        bg_cell_fv_G = [bg_fv_G; cell_fv_G];

        % Input table for classification model
        fv_table_S_G = table(bg_cell_fv_S, bg_cell_fv_G, bg_cell_fv_class, 'VariableNames', {'entropy', 'gradient', 'Label'}); 

        % Gradient feature vector of image to be classified
        test_fv_G = test_img_gradient(1:(rows*cols))'; % feature vector of all pixels

        test_fv_S_G_table = table(test_fv_S, test_fv_G, 'VariableNames', {'entropy', 'gradient'});
        test_fv_table = test_fv_S_G_table;       
    end     
    
    % Apply trained model to image to be classified
    if (entr == true) && (grad == false)
        [trainedClassifier, validationAccuracy] = Bayes_S_trainClassifier(fv_table_S);
    elseif (entr == true) && (grad == true)
        [trainedClassifier, validationAccuracy] = Bayes_S_G_trainClassifier(fv_table_S_G);
    end
    
    yfit = trainedClassifier.predictFcn(test_fv_table);
    p=1;
    for j = 1:cols
        for i = 1:rows
            BW_cube_fv(i,j,k)=strcmp(yfit{p},'cell'); % convert strings back to binary mask image
            p=p+1; 
        end   
    end
    
    % Morphological operations (size filter and smoothing)
    BW_2 = bwareaopen(BW_cube_fv(:,:,k),MinPixelsInConnectedRegion);
    se = strel('disk',SizeOfSmoothingDisk); 
    clzBW = imclose(BW_2,se);
   
    % Check for cells touching the image border and remove.
    % NOTE: Will give error if all segmented cells touch border!
    pre_BW_cube = imfill(clzBW,'holes');
    pre_BW_cube = imclearborder(pre_BW_cube,8); % Comment out if you want to include border touching cells 
    pre_L = bwlabel(pre_BW_cube);
    if sum(sum(pre_L)) == 0
       No_Labels_Error = 'Either no cells found or all are touching the border! Comment out imclearborder() function'    
    end
    
    % Build the binary output images matrix (first image will be blank)
    BW_cube_fv(:,:,k) = pre_BW_cube; 
    k
end
toc

%% Part 5: Write video file; tabulate cell parameters; compute the minimum distance matrix at each time step
NearTh = 100; % Distance threshold in pixels that determines 'close encounters' i.e. green line on segmentations
NF_max = 0;

% Generate file names and objects for output files
idx = strfind(dir_str, filesep);
file_out_name = dir_str(idx(1, size(idx,2))+1:end);
% Video
file_out_name_v = strcat(file_out_name,'.avi');
v = VideoWriter([dir_str(1:idx(size(idx,2))) file_out_name_v]);
v.FrameRate = 2;
open(v);

% Build video file and output parameter table
i = 1;
for k=2:NumOfFrames
    SegFig = figure;
    L = bwlabel(BW_cube_fv(:,:,k));
    stats = regionprops('table',L,'Area','Centroid');
    num_labels = size(stats, 1);
    
    % Tables of parameters. First timepoint table will be blank
    params{k} = stats; % Build up a cell that contains all meta data for exporting and analysis at a later date
    
    B = bwboundaries(BW_cube_fv(:,:,k));
    NF = length(B); % number of cells or cell clusters segmented
    if NF > NF_max
        NF_max = NF;
    end
  
    % Display. 
    imshow(imadjust(I(:,:,k))); colormap bone; hold on;

    for i = 1:length(B)
        bdd = B{i};
        plot(bdd(:,2), bdd(:,1), 'c', 'LineWidth', 1)
    end
       
    % Find minimum distances between boundaries of segmented regions.
    D_min = zeros(NF,NF);
    for i=1:NF-1
        bdd1 = B{i}; n1 = length(bdd1);
        x1 = bdd1(:,2); y1 = bdd1(:,1);       
        for j=i+1:NF           
            bdd2 = B{j}; n2 = length(bdd2);
            x2 = bdd2(:,2); y2 = bdd2(:,1);
            X1 = repmat(x1,[1 n2]); Y1 = repmat(y1,[1 n2]);
            X2 = repmat(x2,[1 n1]); Y2 = repmat(y2,[1 n1]);
            D = round(sqrt( (X1-X2').^2 + (Y1-Y2').^2 ));
            [D_min_in_col,row_vec] = min(D);
            [D_min(i,j),col] = min(D_min_in_col);
            row = row_vec(col);
            if D_min(i,j) < NearTh % Nearest neighbor connecting line
               plot([ bdd1(row,2) bdd2(col,2) ],[ bdd1(row,1) bdd2(col,1) ],'-g')
            end
        end        
    end
    minD{k} = D_min;
   
    pause(0.01); hold off;
    
    F(k) = getframe(gcf);
    writeVideo(v,F(k)); % Saved to folder one level above tiff file folder
    close(SegFig);
end
close(v);
%% Part 6: Save segmentation results in the form of binary images (.mat file) to folder one level above tiff file folder

%save('Declumping_Fig5ab_SSL_output.mat', 'BW_cube_fv');
file_out_name_mat = strcat(file_out_name,'_cell_mask','.mat');
save([dir_str(1:idx(size(idx,2))) file_out_name_mat], 'BW_cube_fv');
formatSpec = 'Video and segmentation mask output files saved to the following folder: ';
str = sprintf(formatSpec)
v.Path
