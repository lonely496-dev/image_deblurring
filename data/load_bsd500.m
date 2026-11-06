function [images, filenames] = load_bsd500(dataDir, setName)


if nargin < 1 || isempty(dataDir)
    dataDir = fullfile(pwd, 'BSDS500-master');
end
if nargin < 2 || isempty(setName)
    setName = 'train';
end
setName = lower(setName);
if ~ismember(setName, {'train', 'val', 'test'})
    error('load_bsd500:badSet', ...
        'Unknown set "%s". Use ''train'', ''val'', or ''test''.', setName);
end

if ~isfolder(dataDir)
    download_and_extract(dataDir);
end

imagesDir = fullfile(dataDir, 'BSDS500', 'data', 'images', setName);
if ~isfolder(imagesDir)
    error('load_bsd500:missingFolder', ...
        'Expected image folder not found: %s', imagesDir);
end

files = dir(fullfile(imagesDir, '*.jpg'));
if isempty(files)
    error('load_bsd500:noImages', ...
        'No .jpg files found in: %s', imagesDir);
end

n = numel(files);
images = cell(n, 1);
filenames = cell(n, 1);
fprintf('Loading %d images from BSDS500 "%s" set...\n', n, setName);
for i = 1:n
    fp = fullfile(imagesDir, files(i).name);
    images{i} = im2double(imread(fp));
    filenames{i} = files(i).name;
end
fprintf('Loaded %d images.\n', n);
end

% ---------------------------------------------------------------------
function download_and_extract(dataDir)
url = 'https://github.com/BIDS/BSDS500/archive/refs/heads/master.zip';
parentDir = fileparts(dataDir);
if isempty(parentDir)
    parentDir = pwd;
end
zipFile = fullfile(parentDir, 'BSDS500-master.zip');

fprintf('BSDS500 not found locally. Downloading (~70 MB)...\n');
try
    websave(zipFile, url);
catch ME
    error('load_bsd500:downloadFailed', ...
        ['Download failed (%s). If MATLAB Online blocks outbound ' ...
        'requests in your environment, download the zip manually ' ...
        'from %s and upload it to MATLAB Drive as %s, then rerun.'], ...
        ME.message, url, zipFile);
end

fprintf('Extracting...\n');
unzip(zipFile, parentDir);   % creates BSDS500-master/ under parentDir

if ~isfolder(dataDir)
    error('load_bsd500:extractFailed', ...
        'Extraction did not produce expected folder: %s', dataDir);
end
end