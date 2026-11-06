%% Setup
scriptDir = fileparts(mfilename('fullpath'));
root = fileparts(scriptDir);
figDir = fullfile(root, 'figures');
if ~isfolder(figDir), mkdir(figDir); end
addpath(fullfile(root, 'data'));
addpath(fullfile(root, 'forward'));
addpath(fullfile(root, 'sgp'));
addpath(fullfile(root, 'baselines'));
addpath(fullfile(root, 'eval'));

[imgs, ~] = load_bsd500(fullfile(root, 'data', 'BSDS500-master'), 'train');
x_true = imgs{1};
psf = build_psf('gaussian', 21, 5);

%% 1. metrics
m_perfect = metrics(x_true, x_true);
m_perfect.psnr
m_perfect.ssim
b_blur = apply_blur(x_true, psf);
m_blur = metrics(b_blur, x_true);
m_blur.psnr
m_blur.ssim

%% 2. run_experiment
opts.max_iter = 100;
results = run_experiment(x_true, psf, 'gaussian', 0.02, opts);

results.sgp.metrics
results.pg.metrics
results.fista.metrics
results.blurred_metrics

figure;
subplot(1,4,1); imshow(results.blurred);   title('blurred+noisy');
subplot(1,4,2); imshow(results.sgp.x);     title('SGP');
subplot(1,4,3); imshow(results.pg.x);      title('plain PG');
subplot(1,4,4); imshow(results.fista.x);   title('FISTA');
saveas(gcf, fullfile(figDir, 'eval_experiment.png'));

%% 3. batch_eval
opts_batch.max_iter = 80;
opts_batch.tol = 0;
summary = batch_eval(5, psf, 'gaussian', 0.02, opts_batch);

summary.blurred_psnr_mean
summary.sgp.psnr_best_mean
summary.pg.psnr_best_mean
summary.fista.psnr_best_mean
summary.sgp.best_iter_mean
summary.pg.best_iter_mean
summary.fista.best_iter_mean