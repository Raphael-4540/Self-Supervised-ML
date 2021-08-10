% Declumping Demo for Self-Supervised Learning Algorithm
% August 2021
%
% This code was tested on Matlab v2020a & v2021a using commercially
% available laptop computers running Windows 10 operating system.
%
% We recommend evaluating the code section by section
% (Ctrl+Enter) and reading the section comments which incorporate step by step 
% instructions and explanations.
%
% Demo image pairs and nuclei masks can be found for testing purposes:
% https://zenodo.org/record/5167318#.YQ_R2ohKhPY
%
% The demo images files are not large, so you can download onto your computer 
% and read into the Matlab workspace using the example dir_str format below.
%
% Run from start to finish, this demo code uses the segmented cells from
% SSL in conjunction with a binary nuclei mask to attribute each pixel
% within a segmented object to an individual nuclei as a strategy for
% declumping. In our manuscript, we described a fluorescent-tag free
% approach (nuclei lensing effect), or simply incorporate a fluorescent
% nuclei tag when taking time-lapse imagery to generate a nuclei mask.
%
% Note: the current code ignores cells touching the boundary. See
% SSL_Demo_2 comments for instrunctions on how to include.
%
% As such, it requires the following inputs of inputs in Sections 1 and 2:
% 1) That the nuclei mask be the same size (i.e. resolution) as the raw
% time-lapse imagery used for the SSL mask.
% 2) The nuclei mask be stored as a binary tiff
% 3) The two consecutive time-lapse images that were used by the SSL code
% 4) The output .mat file of the SSL code
%
% Functions called:
%  ReadDataCube(): used to read in tiff files into a 3D matrix
% 
% Note: each section has step-by-step figures commented out
%% 1. Read in raw imagery data.
% Start fresh
clear; close all; clc; 

% USER INPUT REQUIRED!!
% Choose directory of the two images that were used by SSL to create cell
% masks. e.g.: dir_str = 'C:\Users\Marc\Desktop\Figure5ab_SSL_Imagery\Figure5ab_SSL_Imagery'
dir_str = 'C:\Users\Marc\Desktop\Figure5ab_SSL_Imagery\Figure5ab_SSL_Imagery'

% Read in data
I = ReadDataCube(dir_str);
% Construct I_pair for image overlays
I_pair(:,:,1) = I(:,:,1);
I_pair(:,:,2) = I(:,:,2);

%% 2. Load SSL *.mat cell mask file and read nuclei mask file
%
% % USER INPUT REQUIRED!!
% In this section you will need to place files in your current
% MATLAB directory and change the hardcoded names in the 'load' and
% 'imread' commands

% Place the associated *_cell_mask.mat SSL output file in your current
% MATLAB directory and change the name in the 'load' command
% example: load('Figure5ab_SSL_cell_mask.mat'); 
load('Figure5ab_SSL_Imagery_cell_mask.mat'); % SSL cell mask in your current MATLAB directory

% Place the associated *.tiff binary mask of nuclei in your current
% MATLAB directory and change the name in the 'imread' command
% example: imread('Fig5ab nuclei mask.tif'); 
Mask = imread('Fig5ab nuclei mask.tif'); %binary mask of nucleus in your current MATLAB directory

BW_nucleus = imbinarize(Mask);% convert Mask into binary if not already
B_nucleus = bwboundaries(BW_nucleus); % perimeter of nuclei
B = bwboundaries(BW_cube_fv(:,:,2)); %perimeter of cells

%% 3. Display SSL + Nuclei Boundaries

%  Display cell imagery
figure; imshow(imadjust(I_pair(:,:,2))); colormap bone; hold on;

% Plot SSML boundaries
for i = 1:length(B)
  bdd = B{i};
  plot(bdd(:,2), bdd(:,1), 'c', 'LineWidth', 1)
end

% Plot nucleus boundaries
for i = 1:length(B_nucleus)
  bdd_nucleus = B_nucleus{i};
  plot(bdd_nucleus(:,2), bdd_nucleus(:,1), 'r', 'LineWidth', 1)
end

title(['Nuclei - red ; SSL Segmented Cell Clumps - blue']);   
pause(0.01); hold off;

%% 4. Declumping 

% Rename these arrays.
Ib = BW_cube_fv(:,:,2); 
[nx,ny] = size(Ib);
%figure; imagesc(Ib); colorbar; axis square;

% Apply a size filter to the binary image for nuclei, In.
min_nucleus_size = 10; %originally 250
In = bwareafilt(BW_nucleus,[min_nucleus_size inf]);

% Mask for the "unclaimed" pixels.
ucl_bin_idx = Ib - In;
mask = (ucl_bin_idx >= 0); % Removes -1 values due to "orphan" nuclei.
uidx = mask.*ucl_bin_idx;
% figure; imagesc(uidx); colorbar; axis square;
% figure; imagesc(ucl_bin_idx); colorbar; axis square;

% Form label matrices for connected regions.
Lb = bwlabel(Ib);
Ln = bwlabel(In);
% figure; imagesc(Lb); colormap('colorcube'); colorbar; axis square;
% figure; imagesc(Ln); colormap('colorcube'); colorbar; axis square;

% Form the boundary matrices.
%[B_b,L_b] = bwboundaries(Ib,'noholes');
[B_nc,L_nc] = bwboundaries(mask.*In,'noholes');
Lnc = bwlabel(mask.*In);
% figure; imagesc(Lnc); colormap('colorcube'); colorbar; axis square;

% Map every label in 'Ln' to a label in 'Lb'.
n_Lb = max(unique(Lb));
n_Lnc = max(unique(Lnc));
Lb_idx = zeros(n_Lnc,1);
for i=1:n_Lnc
    select = (Lnc==i).*Lb;
    Lb_idx(i) = max(unique(select));
end
Lnc_idx = (1:n_Lnc)';
%disp('Map Nucleus labels to OF labels')
%[ Lnc_idx  Lb_idx ]

% Create a cell array for each 'Lb' region showing the constituent 'Ln'
% regions within it.
% First, sort by the Lb_idx in ascending order.
[out,idx] = sort(Lb_idx);
a = [ out Lnc_idx(idx) ];
% Second, assign 'Ln' regions to each 'Lb' region.
LnInLb_cnts = zeros(n_Lb,1);
LnInLb = cell(n_Lb,1);
for i=1:n_Lb
    idx = ( a(:,1) == i );
    LnInLb_cnts(i) = sum(idx);
    LnInLb{i,1} = a(idx,2);
end

% Form a cell array that contains the unclaimed pixels locations for each Lb.
%[X,Y] = meshgrid(1:nx,1:ny); 
[X,Y] = meshgrid(1:ny,1:nx);
Xvec = X(:); Yvec = Y(:);
unclaimed_pixels_cnts = zeros(n_Lb,1);
unclaimed_pixels = cell(n_Lb,1);
for i=1:n_Lb
    % Find all the "unclaimed" pixels in that region.
    unclaimed_pixels_idx = logical((Lb==i).*uidx);
    unclaimed_pixels{i} = [Xvec(unclaimed_pixels_idx(:)) Yvec(unclaimed_pixels_idx(:))];
    [unclaimed_pixels_cnts(i),~] = size(unclaimed_pixels{i});
end
%disp('     OF Label   # of unclaimed pixels  # of Nucl Labels  ');
%[ (1:n_Lb)' unclaimed_pixels_cnts LnInLb_cnts ]

% Assign a nucleus label to each unclaimed pixel.
unclaimed_pixel_label = cell(n_Lb,1);
for i=1:n_Lb
    b = LnInLb{i}; % Indices of nucleus boundaries contained Lb region 'i'.
    Xuncl = unclaimed_pixels{i}(:,1);
    Yuncl = unclaimed_pixels{i}(:,2);
    dist = zeros(unclaimed_pixels_cnts(i),LnInLb_cnts(i));
    for k=1:LnInLb_cnts(i) % Loop through the nuclei in the 'i'th Lb region.
        Xbdd = B_nc{b(k)}(:,1);
        Ybdd = B_nc{b(k)}(:,2);
        [nb,~] = size(Xbdd);
        
        % Compute the x and y difference between unclaimed pixels and
        % boundary pixels.
        %Xdiff = repmat(Xuncl,[1 nb]) - repmat(Xbdd',[unclaimed_pixels_cnts(i) 1]);
        %Ydiff = repmat(Yuncl,[1 nb]) - repmat(Ybdd',[unclaimed_pixels_cnts(i) 1]);
        % NOTE the use of Yuncl with Xbdd and Xuncl with Ybdd.
        Xdiff = repmat(Yuncl,[1 nb]) - repmat(Xbdd',[unclaimed_pixels_cnts(i) 1]);
        Ydiff = repmat(Xuncl,[1 nb]) - repmat(Ybdd',[unclaimed_pixels_cnts(i) 1]);
        
        % Compute the Euclidean distance between each unclaimed pixel and
        % the boundary pixels of k-th nucleus region.  Find the minimum
        % distance.
        dist(:,k) = min((Xdiff.^2 + Ydiff.^2),[],2);
    end
    % Find the row index of the nucleus region with the closest boundary
    % pixel to each unclaimed pixel in the ith Lb region.
    [m,idx] = min(dist,[],2);
    % Convert convert the row index 'idx' to the nuclei label index.
    unclaimed_pixel_label_tmp = zeros(unclaimed_pixels_cnts(i),1);
    for j=1:unclaimed_pixels_cnts(i)
        unclaimed_pixel_label_tmp(j) = b(idx(j));
    end
    unclaimed_pixel_label{i} = unclaimed_pixel_label_tmp;
end

% Now, assemble a complete label image using the NN assignments above.
L_draft = zeros(nx,ny); 
for i=1:n_Lb
    % Initialize temporary storage.
    L_temp = zeros(nx,ny);
    I_temp_vec = L_temp(:); 
    % Determine the logical indices of the unclaimed pixels in the i-th Lb region.
    unclaimed_pixels_idx = logical((Lb==i).*uidx);
    unclaimed_pixels_idx_vec = unclaimed_pixels_idx(:);
    c = unclaimed_pixel_label{i}(:,1);
    I_temp_vec(unclaimed_pixels_idx_vec') = c;
    L_temp = reshape(I_temp_vec,[nx ny]);
    L_draft = L_draft + L_temp;
end
% Combine with the pixels directly assigned labels via the nucleus lensing.
L_draft = L_draft + L_nc; 
% Display.
% figure; imagesc(L_draft); axis square; colormap(colorcube); colorbar;
% title('Declumped Cells using Nucleus Lensing')

% Fix the "misfit" regions due to Euclidean distance from nucleus boundary.
% Edit the final label assignments by removing the minor disconnected
% regions that have been mislabeled.

% Initialize binary image for "misfit" regions.
L_misfits_total = zeros(nx,ny);
for k=1:n_Lnc
    % Select out k-th nucleus-identified bio-cell.
    I_local = (L_draft == k).*L_draft;
    % Label the disconnected regions.
    L = bwlabel(I_local);
    % Find number of disconnected pieces to this bio-cell region.
    NumOfRegs = size(unique(L),1)-1;
    % If there is more than one region then the label has been mis-assigned.
    if NumOfRegs > 1
        % Find the area of the disconnected regions.
        area = zeros(NumOfRegs,1);
        for j=1:NumOfRegs
            area(j) = sum(sum( L==j  ));
        end
        k
        % Assume that the largest area is the region correctly assigned.
        % Find the index in the L image corresponding to the largest area.
        [~,idx] = max(area);
        % Set the largest area
        % Set the largest area to label=0 for the "misfits" label image.
        L_misfits = (~(L == idx)).*L;
        % Set the smaller areas to label=1.
        L_misfits = (L_misfits>0);
        % Combine with previously detected misfits from other regions.
        L_misfits_total = L_misfits_total + L_misfits;
    end
end

% Assign misfits to background.
L_draft_nomisfits =(~L_misfits_total).*L_draft; 
% Display.
% figure; imagesc(L_draft_nomisfits); axis square; colormap(colorcube); colorbar;
% title('Declumped Cells using Nucleus Lensing (DEL misfits)')

% Assemble together all the boundaries of the de-clumped cells.
tic
BL = cell(n_Lnc,1);
for k=1:n_Lnc
    [bdd,~] = bwboundaries( (L_draft_nomisfits == k).*L_draft_nomisfits );
    BL{k} = bdd{1}(:,:);
end
toc

% Display draft segmentation.
figure; imshow(imadjust(I_pair(:,:,2))); colormap bone; hold on;
% Plot SSML boundaries
for k = 1:n_Lnc
  bdd = BL{k};
  plot(bdd(:,2), bdd(:,1), 'b', 'LineWidth', 1)
end

title(['Declumped Cell Results']);   
pause(0.01); hold off;

% Plot nucleus boundaries
% for i = 1:length(B_nucleus)
%   bdd_nucleus = B_nucleus{i};
%   plot(bdd_nucleus(:,2), bdd_nucleus(:,1), 'r', 'LineWidth', 1)
% end










