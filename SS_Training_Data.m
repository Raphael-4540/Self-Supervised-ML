function [bg_train_entr, cell_train_entr] = SS_Training_Data(I_pair,MinPixelsInConnectedRegion, S_nhood, extra_S)
% Image pair is an input for optical flow algorithm to isolate background and
% cells using entropy measurements for self-tuning the threshold
SizeOfSmoothingDisk = 5; 

% Entropy image for self-supervised thresholding
train_img_entropy = entropyfilt(I_pair(:,:,1), S_nhood);

% Generate Background Training Data.
bg_th_init = 1e-4; % Good initial setting for over exposing cells (1e-4 default)
bg_train_S_total = 1; % initialize the cumulative S value
while bg_train_S_total <= 1e5 
    % Compute optical flow segmentation.
    [BW_Low_th(:,:,1), mag] = OpticalFlowSegmentation(I_pair,bg_th_init,MinPixelsInConnectedRegion,SizeOfSmoothingDisk);
    bg_mag = ~BW_Low_th(:,:,1).*mag; % remove cells

	% Use the OF mag() images to mask original tiff and create training
    % data for the background
    bg_train_entr = (double(logical(bg_mag))).*train_img_entropy; % OF segmented background (raw image)
    bg_train_S_total = sum(sum(bg_train_entr))
    if bg_th_init > 0.002
        break;
    end
    bg_th_init = bg_th_init + 1e-4;
end

    
% Generate Cell Training Data
cell_train_S_total = 1; % initialize the cumulative S value
i = 1;
S_total_th = 1e5; 
cell_th_init = 0.04;
while cell_train_S_total <= S_total_th 
    
    if cell_th_init <= 0 % If this has gone negative a significant amount of entropy has not been found
        cell_th_init = 0.00002 % default th = 0.00002: Try and find something to train on.
        [BW_high_th(:,:,1), mag] = OpticalFlowSegmentation(I_pair,cell_th_init,MinPixelsInConnectedRegion,SizeOfSmoothingDisk);
        cell_mag = BW_high_th(:,:,1).*mag; 
        cell_train_entr = (double(logical(cell_mag))).*train_img_entropy;
        break;
    end
    
    [BW_high_th(:,:,1), mag] = OpticalFlowSegmentation(I_pair,cell_th_init,MinPixelsInConnectedRegion,SizeOfSmoothingDisk);
    cell_mag = BW_high_th(:,:,1).*mag; % show cells
    if (i==1)
        cell_train_entr = (double(logical(cell_mag))).*train_img_entropy; 
        cell_train_S_total = sum(sum(cell_train_entr)) 
        S_total_th = cell_train_S_total + extra_S; 
    else
        cell_train_entr = (double(logical(cell_mag))).*train_img_entropy; % OF segmented background (raw image)
        cell_train_S_total = sum(sum(cell_train_entr))
    end
    cell_th_init = cell_th_init - 2e-3
    i = i + 1;
end


end