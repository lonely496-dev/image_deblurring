%% Setup
scriptDir = fileparts(mfilename('fullpath'));
root = fileparts(scriptDir);
addpath(fullfile(root, 'data'));
addpath(fullfile(root, 'forward'));
addpath(scriptDir);

[imgs, ~] = load_bsd500(fullfile(root, 'data', 'BSDS500-master'), 'train');
x_true = imgs{1};

%% 1. Identical image - should score "perfect"
m_perfect = metrics(x_true, x_true);
m_perfect.psnr    % expect Inf (or very large)
m_perfect.ssim    % expect 1.0
m_perfect.relerr  % expect 0

%% 2. Degraded image - should score worse
psf = build_psf('gaussian', 15, 4);
b = apply_blur(x_true, psf);
m_blurred = metrics(b, x_true);
m_blurred.psnr    % expect noticeably < Inf, finite
m_blurred.ssim    % expect < 1.0
m_blurred.relerr  % expect > 0

%% 3. Sanity - blurring more should score worse than blurring less
psf_mild = build_psf('gaussian', 7, 1);
psf_heavy = build_psf('gaussian', 21, 6);
m_mild = metrics(apply_blur(x_true, psf_mild), x_true);
m_heavy = metrics(apply_blur(x_true, psf_heavy), x_true);
m_mild.psnr > m_heavy.psnr   % expect 1 (mild blur = higher PSNR = better)
m_mild.ssim > m_heavy.ssim   % expect 1