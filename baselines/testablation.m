scriptDir = fileparts(mfilename('fullpath'));
addpath(scriptDir);
addpath(fullfile(fileparts(scriptDir), 'sgp'));
addpath(fullfile(fileparts(scriptDir), 'forward'));
addpath(fullfile(fileparts(scriptDir), 'data'));

[imgs, ~] = load_bsd500(fullfile(fileparts(scriptDir), 'data', 'BSDS500-master'), 'train');
x_true = imgs{1};

psf = build_psf('gaussian', 21, 5);
bx = apply_blur(x_true, psf);
b = add_noise(bx, 'gaussian', 0.02);

opts.max_iter = 300;
opts.bb_rule = 'alternate';
opts.tol = 0;   % force full iteration budget, fair comparison

[x_sgp, info_sgp] = sgp_solve(@grad_ls, b, psf, b, opts);
[x_pg,  info_pg]  = projected_gradient(@grad_ls, b, psf, b, opts);

info_sgp.iters
info_pg.iters
info_sgp.f(end)
info_pg.f(end)

figure;
plot(info_sgp.f, 'LineWidth', 1.5); hold on;
plot(info_pg.f, 'LineWidth', 1.5);
legend('SGP (scaled)', 'Plain projected gradient');
xlabel('iteration'); ylabel('f(x_k)');
title('Scaling ablation (LS, full budget)'); grid on;

figure;
subplot(1,3,1); imshow(x_true); title('original');
subplot(1,3,2); imshow(x_sgp);  title('SGP');
subplot(1,3,3); imshow(x_pg);   title('plain PG');