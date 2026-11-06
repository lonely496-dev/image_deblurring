%% Setup
scriptDir = fileparts(mfilename('fullpath'));
root = fileparts(scriptDir);
addpath(fullfile(root, 'data'));
addpath(fullfile(root, 'forward'));
addpath(fullfile(root, 'sgp'));
addpath(fullfile(root, 'baselines'));
addpath(scriptDir);

[imgs, ~] = load_bsd500(fullfile(root, 'data', 'BSDS500-master'), 'train');
x_true = imgs{1};
psf = build_psf('gaussian', 21, 5);

%% Run one trial, Gaussian noise
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