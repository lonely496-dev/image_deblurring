function summary = batch_eval(numImages, psf, noiseType, noiseParam, opts)
% Runs run_experiment in the first numImages BSD500 training images,
% aggregates best-achieved PSNR/SSIM and their iteration counts per method.

root = fileparts(fileparts(mfilename('fullpath')));
[imgs, ~] = load_bsd500(fullfile(root, 'data', 'BSDS500-master'), 'train');
numImages = min(numImages, numel(imgs));

methods = {'sgp', 'pg', 'fista'};
for m = 1:numel(methods)
    agg.(methods{m}).psnr_final = zeros(numImages, 1);
    agg.(methods{m}).ssim_final = zeros(numImages, 1);
    agg.(methods{m}).psnr_best  = zeros(numImages, 1);
    agg.(methods{m}).ssim_best  = zeros(numImages, 1);
    agg.(methods{m}).best_iter  = zeros(numImages, 1);
end
agg.blurred_psnr = zeros(numImages, 1);

for i = 1:numImages
    rng(i);
    fprintf('Image %d/%d...\n', i, numImages);
    x_true = imgs{i};
    results = run_experiment(x_true, psf, noiseType, noiseParam, opts);

    agg.blurred_psnr(i) = results.blurred_metrics.psnr;
    for m = 1:numel(methods)
        r = results.(methods{m});
        agg.(methods{m}).psnr_final(i) = r.metrics.psnr;
        agg.(methods{m}).ssim_final(i) = r.metrics.ssim;
        agg.(methods{m}).psnr_best(i)  = r.metrics_best.psnr;
        agg.(methods{m}).ssim_best(i)  = r.metrics_best.ssim;
        agg.(methods{m}).best_iter(i)  = r.info.best_iter;
    end
end

summary.blurred_psnr_mean = mean(agg.blurred_psnr);
for m = 1:numel(methods)
    mm = methods{m};
    summary.(mm).psnr_final_mean = mean(agg.(mm).psnr_final);
    summary.(mm).psnr_final_std  = std(agg.(mm).psnr_final);
    summary.(mm).ssim_final_mean = mean(agg.(mm).ssim_final);
    summary.(mm).psnr_best_mean  = mean(agg.(mm).psnr_best);
    summary.(mm).psnr_best_std   = std(agg.(mm).psnr_best);
    summary.(mm).ssim_best_mean  = mean(agg.(mm).ssim_best);
    summary.(mm).best_iter_mean  = mean(agg.(mm).best_iter);
    summary.(mm).best_iter_std   = std(agg.(mm).best_iter);
end
summary.raw = agg;
end