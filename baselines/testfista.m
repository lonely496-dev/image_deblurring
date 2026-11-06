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
opts.tol = 0;
opts.bb_rule = 'alternate';

[x_sgp, info_sgp] = sgp_solve(@grad_ls, b, psf, b, opts);
[x_pg,  info_pg]  = projected_gradient(@grad_ls, b, psf, b, opts);
[x_fista, info_fista] = fista_solve(@grad_ls, b, psf, b, opts);

info_sgp.f(end)
info_pg.f(end)
info_fista.f(end)

figure;
plot(info_sgp.f, 'LineWidth', 1.5); hold on;
plot(info_pg.f, 'LineWidth', 1.5);
plot(info_fista.f, 'LineWidth', 1.5);
legend('SGP', 'Plain PG', 'FISTA');
xlabel('iteration'); ylabel('f(x_k)');
title('Algorithm comparison (LS objective)'); grid on;

figure;
subplot(1,4,1); imshow(x_true);   title('original');
subplot(1,4,2); imshow(x_sgp);    title('SGP');
subplot(1,4,3); imshow(x_pg);     title('plain PG');
subplot(1,4,4); imshow(x_fista);  title('FISTA');