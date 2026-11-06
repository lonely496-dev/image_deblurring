scriptDir = fileparts(mfilename('fullpath'));
root = fileparts(scriptDir);
figDir = fullfile(root, 'figures');
resultsDir = fullfile(root, 'results');
if ~isfolder(figDir), mkdir(figDir); end
if ~isfolder(resultsDir), mkdir(resultsDir); end
addpath(fullfile(root, 'data'));
addpath(fullfile(root, 'forward'));
addpath(fullfile(root, 'sgp'));
addpath(fullfile(root, 'baselines'));
addpath(fullfile(root, 'eval'));

psf = build_psf('gaussian', 21, 5);
opts.max_iter = 200;
opts.tol = 0;
opts.bb_rule = 'alternate';

summary = batch_eval(30, psf, 'gaussian', 0.02, opts);
for o = {'pg','fista'}
    d = summary.raw.sgp.psnr_best - summary.raw.(o{1}).psnr_best;
    ci = 1.96 * std(d) / sqrt(numel(d));
    fprintf('SGP - %s best PSNR: %.3f dB (95%% CI +/- %.3f)\n', o{1}, mean(d), ci);
    d = summary.raw.sgp.psnr_final - summary.raw.(o{1}).psnr_final;
    ci = 1.96 * std(d) / sqrt(numel(d));
    fprintf('SGP - %s final PSNR: %.3f dB (95%% CI +/- %.3f)\n', o{1}, mean(d), ci);
end

save(fullfile(resultsDir, 'batch_summary.mat'), 'summary');

methods = {'sgp', 'pg', 'fista'};
labels = {'SGP', 'Plain PG', 'FISTA'};
colors = lines(3);

means = zeros(1,3); stds = zeros(1,3);
for m = 1:3
    means(m) = summary.(methods{m}).psnr_best_mean;
    stds(m)  = summary.(methods{m}).psnr_best_std;
end
figure;
bar(means); hold on;
errorbar(1:3, means, stds, 'k.', 'LineWidth', 1.2);
yline(summary.blurred_psnr_mean, '--r', 'Blurred (no restoration)');
set(gca, 'XTickLabel', labels);
ylabel('Best-achieved PSNR (dB)');
title('Restoration quality 30 BSD500 images');
grid on;
saveas(gcf, fullfile(figDir, 'full_eval_psnr_bar.png'));

iters = zeros(1,3); iters_std = zeros(1,3);
for m = 1:3
    iters(m) = summary.(methods{m}).best_iter_mean;
    iters_std(m) = summary.(methods{m}).best_iter_std;
end
figure;
bar(iters); hold on;
errorbar(1:3, iters, iters_std, 'k.', 'LineWidth', 1.2);
set(gca, 'XTickLabel', labels);
ylabel('Iterations to reach best PSNR');
title('Speed to peak quality (semiconvergence check)');
grid on;
saveas(gcf, fullfile(figDir, 'full_eval_best_iter_bar.png'));

ssim_means = zeros(1,3);
for m = 1:3
    ssim_means(m) = summary.(methods{m}).ssim_best_mean;
end
figure;
bar(ssim_means);
set(gca, 'XTickLabel', labels);
ylabel('Best-achieved SSIM');
title('Structural similarity across 30 BSD500 images');
grid on;
saveas(gcf, fullfile(figDir, 'full_eval_ssim_bar.png'));

figure;
for m = 1:3
    subplot(1,3,m);
    scatter(summary.raw.(methods{m}).psnr_final, summary.raw.(methods{m}).psnr_best, 40, colors(m,:), 'filled');
    hold on;
    lims = [min(summary.raw.(methods{m}).psnr_final), max(summary.raw.(methods{m}).psnr_best)];
    plot(lims, lims, 'k--');
    xlabel('PSNR at final iteration'); ylabel('Best PSNR achieved');
    title(labels{m}); grid on; axis equal;
end
sgtitle('Final vs. best PSNR per image (gap = semiconvergence loss)');
saveas(gcf, fullfile(figDir, 'full_eval_final_vs_best_scatter.png'));

summary.blurred_psnr_mean
summary.sgp.psnr_best_mean
summary.pg.psnr_best_mean
summary.fista.psnr_best_mean
summary.sgp.best_iter_mean
summary.pg.best_iter_mean
summary.fista.best_iter_mean