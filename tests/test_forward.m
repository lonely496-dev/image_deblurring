%% Setup
scriptDir = fileparts(mfilename('fullpath'));
root = fileparts(scriptDir);

figDir = fullfile(root, 'figures');
if ~isfolder(figDir), mkdir(figDir); end
addpath(fullfile(root, 'data'));
addpath(fullfile(root, 'forward'));

[imgs, ~] = load_bsd500(fullfile(root, 'data', 'BSDS500-master'), 'train');
x_true = imgs{1};

%% 1. load_bsd500 - shape/type sanity
numel(imgs)          % expect 200
size(imgs{1})
class(imgs{1})

%% 2. build_psf - kernel shape and normalization
psf_g = build_psf('gaussian', 7, 1.5);
psf_m = build_psf('motion', 9, [7, 30]);
sum(psf_g(:))        % expect ~1.0
sum(psf_m(:))         % expect ~1.0
figure;
subplot(1,2,1); imagesc(psf_g); axis image; colorbar; title('Gaussian PSF');
subplot(1,2,2); imagesc(psf_m); axis image; colorbar; title('Motion PSF');
saveas(gcf, fullfile(figDir, 'forward_psf.png'));

%% 3. apply_blur - adjoint identity check (H and H')
psf = build_psf('gaussian', 15, 4);
u = randn(size(x_true));
v = randn(size(x_true));
lhs = sum(apply_blur(u, psf) .* v, 'all');
rhs = sum(u .* apply_blur(v, psf, 'transpose'), 'all');
adjoint_check = abs(lhs - rhs)   % expect ~1e-14

%% 4. add_noise - visual check, both models
bx = apply_blur(x_true, psf);
b_gauss = add_noise(bx, 'gaussian', 0.03);
b_pois  = add_noise(bx, 'poisson', 60);
figure;
subplot(1,4,1); imshow(x_true);  title('original');
subplot(1,4,2); imshow(bx);      title('blurred');
subplot(1,4,3); imshow(b_gauss); title('+gaussian');
subplot(1,4,4); imshow(b_pois);  title('+poisson');
saveas(gcf, fullfile(figDir, 'forward_noise.png'));