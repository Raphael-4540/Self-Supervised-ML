%% Read in a data cube for video microscopy data.
function out = ReadDataCube(dir_str)

files = dir(dir_str);
NumOfFiles = numel(files);

% Read in data from files.
k = 0; % Counter for storage arrays.
for i=3:NumOfFiles % Start from '3' because of '.' and '..' files. (Will vary with OS!!)
    
    % Form string for filename.
    file = [ dir_str '\' files(i).name ];
    
    % Read file to image, map to grayscale and rescale image.
    A = imread(file);
    
    % If RGB then make Grayscale.
    if size(A,3) == 3
        A = rgb2gray(A);
    end
    % If not uint8 then recast.
    if isa(A,'uint8') == false
        A = im2uint8(A);
        %A = im2int16(A);
    end
    
    % Store the reflection and transmission images in arrays.
    k = k + 1;
    out(:,:,k) = A;
    
end

end

