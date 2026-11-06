%% Setup
scriptDir = fileparts(mfilename('fullpath'));
root = fileparts(scriptDir);

figDir = fullfile(root, 'figures');
if ~isfolder(figDir), mkdir(figDir); end
addpath(fullfile(root, 'data'));
addpath(fullfile(root, 'forward'));
addpath(fullfile(root, 'sgp'));
addpath(fullfile(root, 'baselines'));

[imgs, ~] = load_bsd500(fullfile(root, 'data', 'BSDS500-master'), 'train');
x_true = imgs{1};
psf = build_psf('gaussian', 21, 5);
bx = apply_blur(x_true, psf);
b = add_noise(bx, 'gaussian', 0.02);
opts.max_iter = 300; opts.tol = 0; opts.bb_rule = 'alternate';

%% 1. Ablation - SGP vs plain PG, least squares
[x_sgp, info_sgp] = sgp_solve(@grad_ls, b, psf, b, opts);
[x_pg, info_pg]   = projected_gradient(@grad_ls, b, psf, b, opts);
figure;
plot(info_sgp.f, 'LineWidth', 1.5); hold on;
plot(info_pg.f, 'LineWidth', 1.5);
legend('SGP', 'Plain PG'); xlabel('iteration'); ylabel('f(x_k)');
title('Ablation - LS objective'); grid on;
saveas(gcf, fullfile(figDir, 'comparison_ablation_ls.png'));

%% 2. Ablation - SGP vs plain PG, KL/Poisson
b_kl = max(add_noise(bx, 'poisson', 60), 1e-6);
[x_sgp_kl, info_sgp_kl] = sgp_solve(@grad_kl, b_kl, psf, b_kl, opts);
[x_pg_kl, info_pg_kl]   = projected_gradient(@grad_kl, b_kl, psf, b_kl, opts);
figure;
plot(info_sgp_kl.f, 'LineWidth', 1.5); hold on;
plot(info_pg_kl.f, 'LineWidth', 1.5);
legend('SGP', 'Plain PG'); xlabel('iteration'); ylabel('f(x_k)');
title('Ablation - KL/Poisson objective'); grid on;
saveas(gcf, fullfile(figDir, 'comparison_ablation_kl.png'));

%% 3. FISTA comparison (LS)
[x_fista, info_fista] = fista_solve(@grad_ls, b, psf, b, opts);
figure;
plot(info_sgp.f, 'LineWidth', 1.5); hold on;
plot(info_pg.f, 'LineWidth', 1.5);
plot(info_fista.f, 'LineWidth', 1.5);
legend('SGP', 'Plain PG', 'FISTA');
xlabel('iteration'); ylabel('f(x_k)'); title('Algorithm comparison (LS)'); grid on;
saveas(gcf, fullfile(figDir, 'comparison_fista_curves.png'));
figure;
subplot(1,4,1); imshow(x_true);   title('original');
subplot(1,4,2); imshow(x_sgp);    title('SGP');
subplot(1,4,3); imshow(x_pg);     title('plain PG');
subplot(1,4,4); imshow(x_fista);  title('FISTA');
saveas(gcf, fullfile(figDir, 'comparison_fista_images.png'));

%% 4. Semiconvergence diagnostic (PSNR vs objective over iterations, FISTA)
checkpoints = [5 10 20 40 80 150 220 300];
psnr_fista = zeros(size(checkpoints));
f_fista = zeros(size(checkpoints));
for i = 1:numel(checkpoints)
    o.max_iter = checkpoints(i); o.tol = 0;
    [x_k, info_k] = fista_solve(@grad_ls, b, psf, b, o);
    psnr_fista(i) = psnr(x_k, x_true);
    f_fista(i) = info_k.f(end);
end
figure;
subplot(1,2,1); plot(checkpoints, psnr_fista, '-o');
xlabel('iteration'); ylabel('PSNR vs true image'); title('FISTA: PSNR'); grid on;
subplot(1,2,2); plot(checkpoints, f_fista, '-o');
xlabel('iteration'); ylabel('f(x_k)'); title('FISTA: objective'); grid on;
saveas(gcf, fullfile(figDir, 'comparison_semiconvergence.png'));