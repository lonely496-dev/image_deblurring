function psf = build_psf(type, sz, param)
% BUILD_PSF  Generate a point spread function (blur kernel) for the
% forward model b = H*x + n used throughout the SGP experiments.
%
%   psf = BUILD_PSF('gaussian', sz, sigma)
%       sz    : kernel size, scalar (sz x sz) or [rows cols]
%       sigma : standard deviation of the Gaussian
%
%   psf = BUILD_PSF('motion', sz, [len, theta])
%       sz    : kernel size, scalar (sz x sz) or [rows cols]
%       len   : length of the motion blur, in pixels
%       theta : angle of motion, in degrees (0 = horizontal)
%
% OUTPUT
%   psf : kernel normalized so sum(psf(:)) == 1 (preserves image energy)
%
% NOTES
%   - Wraps fspecial (Image Processing Toolbox), which must be on path.
%   - Keep sz small relative to the 32x32 CIFAR-10 images (e.g. 5-9),
%     otherwise the blur destroys nearly all image structure.
%
% USAGE
%   psf1 = build_psf('gaussian', 7, 1.5);
%   psf2 = build_psf('motion', 9, [7, 30]);
%   figure; imagesc(psf1); axis image; colorbar; title('Gaussian PSF');

if nargin < 2 || isempty(sz)
    sz = 7;
end

switch lower(type)
    case 'gaussian'
        if nargin < 3 || isempty(param)
            param = 1.5;   % default sigma
        end
        sigma = param;
        psf = fspecial('gaussian', sz, sigma);

    case 'motion'
        if nargin < 3 || isempty(param)
            param = [7, 0];   % default: length 7, horizontal
        end
        len = param(1);
        theta = param(2);
        psf = fspecial('motion', len, theta);

        % fspecial('motion',...) can return a kernel bigger than sz
        % in one dimension; pad/crop to the requested sz for
        % consistency with apply_blur.m's padding logic.
        psf = fit_to_size(psf, sz);

    otherwise
        error('build_psf:badType', ...
            'Unknown PSF type "%s". Use ''gaussian'' or ''motion''.', type);
end

psf = psf / sum(psf(:));   % ensure energy-preserving normalization
end

% ---------------------------------------------------------------------
function out = fit_to_size(psf, sz)
% Pad or crop psf (centered) to sz x sz (or sz(1) x sz(2)).
if isscalar(sz)
    targetR = sz; targetC = sz;
else
    targetR = sz(1); targetC = sz(2);
end

[r, c] = size(psf);
out = zeros(targetR, targetC);

% Crop if larger
if r > targetR
    startR = floor((r - targetR)/2) + 1;
    psf = psf(startR:startR+targetR-1, :);
    r = targetR;
end
if c > targetC
    startC = floor((c - targetC)/2) + 1;
    psf = psf(:, startC:startC+targetC-1);
    c = targetC;
end

% Place (possibly smaller) psf centered into the target canvas
offR = floor((targetR - r)/2);
offC = floor((targetC - c)/2);
out(offR+1:offR+r, offC+1:offC+c) = psf;
end