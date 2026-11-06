%% Setup
scriptDir = fileparts(mfilename('fullpath'));
root = fileparts(scriptDir);

figDir = fullfile(root, 'figures');
if ~isfolder(figDir), mkdir(figDir); end
addpath(fullfile(root, 'data'));
addpath(fullfile(root, 'forward'));
addpath(fullfile(root, 'sgp'));

[imgs, ~] = load_bsd500(fullfile(root, 'data', 'BSDS500-master'), 'train');
x_true = imgs{1};
psf = build_psf('gaussian', 7, 1.5);
bx = apply_blur(x_true, psf);
b = add_noise(bx, 'gaussian', 0.02);

%% 1. grad_ls - finite-difference check
[f, g] = grad_ls(x_true, b, psf);
idx = randperm(numel(x_true), 5);
fd = zeros(1,5); analytic = zeros(1,5);
for k = 1:5
    xp = x_true; xp(idx(k)) = xp(idx(k)) + 1e-5;
    xm = x_true; xm(idx(k)) = xm(idx(k)) - 1e-5;
    [fp,~] = grad_ls(xp, b, psf);
    [fm,~] = grad_ls(xm, b, psf);
    fd(k) = (fp - fm) / (2e-5);
    analytic(k) = g(idx(k));
end
max(abs(fd - analytic))   % expect ~1e-9

%% 2. grad_kl - finite-difference check
b_kl = max(add_noise(bx, 'poisson', 60), 1e-6);
[f, g] = grad_kl(x_true, b_kl, psf);
idx = randperm(numel(x_true), 5);
fd = zeros(1,5); analytic = zeros(1,5);
for k = 1:5
    xp = x_true; xp(idx(k)) = xp(idx(k)) + 1e-5;
    xm = x_true; xm(idx(k)) = xm(idx(k)) - 1e-5;
    [fp,~] = grad_kl(xp, b_kl, psf);
    [fm,~] = grad_kl(xm, b_kl, psf);
    fd(k) = (fp - fm) / (2e-5);
    analytic(k) = g(idx(k));
end
max(abs(fd - analytic))   % expect ~1e-8

%% 3. scaling_matrix - positivity/bounds sanity
[~, ~, Vplus] = grad_ls(x_true, b, psf);
D = scaling_matrix(x_true, Vplus);
all(D(:) > 0)             % expect 1
any(isnan(D(:))) || any(isinf(D(:)))   % expect 0

%% 4. bb_steplength - known closed-form quadratic check
c = 3;
xp1 = randn(50,1); xc1 = randn(50,1);
gp1 = c*xp1; gc1 = c*xc1;
alpha = bb_steplength(xc1-xp1, gc1-gp1, ones(50,1), 'alternate');
abs(alpha - 1/c)          % expect ~0

%% 5. armijo_linesearch - decrease check
[f_x, g_x] = grad_ls(x_true, b, psf);
dx = -g_x;
fun = @(xx) grad_ls(xx, b, psf);
[lambda, x_new, f_new] = armijo_linesearch(fun, x_true, f_x, g_x, dx);
f_new < f_x                % expect 1

%% 6. sgp_solve - full end-to-end run
psf2 = build_psf('gaussian', 21, 5);
bx2 = apply_blur(x_true, psf2);
b2 = add_noise(bx2, 'gaussian', 0.02);
opts.max_iter = 300; opts.tol = 0; opts.bb_rule = 'alternate';
[x_sgp, info_sgp] = sgp_solve(@grad_ls, b2, psf2, b2, opts);
info_sgp.f(end) < info_sgp.f(1)   % expect 1
figure;
subplot(1,3,1); imshow(x_true); title('original');
subplot(1,3,2); imshow(b2);     title('blurred+noisy');
subplot(1,3,3); imshow(x_sgp);  title('SGP restored');
saveas(gcf, fullfile(figDir, 'sgp_core_restoration.png'));