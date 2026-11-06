function results = run_experiment(x_true, psf, noiseType, noiseParam, opts)
% Runs SGP, plain PG, and FISTA on one blurred/noisy trial of x_true.
% Tracks PSNR per iteration so each method's best-achievable quality
    if nargin < 5
        opts = struct();
    end
    if ~isfield(opts, 'max_iter'), opts.max_iter = 300; end
    if ~isfield(opts, 'tol'), opts.tol = 0; end
    if ~isfield(opts, 'bb_rule'), opts.bb_rule = 'alternate'; end
    opts.x_true = x_true;

    bx = apply_blur(x_true, psf);
    b = add_noise(bx, noiseType, noiseParam);

    if strcmpi(noiseType, 'poisson')
        b = max(b, 1e-6);
        gradFun = @grad_kl;
    else
        gradFun = @grad_ls;
    end

    [x_sgp, info_sgp] = sgp_solve(gradFun, b, psf, b, opts);
    [x_pg,  info_pg]  = projected_gradient(gradFun, b, psf, b, opts);
    [x_fista, info_fista] = fista_solve(gradFun, b, psf, b, opts);

    results.noiseType = noiseType;
    results.blurred = b;

    results.sgp.x = x_sgp;
    results.sgp.info = info_sgp;
    results.sgp.metrics = metrics(x_sgp, x_true);
    results.sgp.metrics_best = metrics(info_sgp.x_best, x_true);

    results.pg.x = x_pg;
    results.pg.info = info_pg;
    results.pg.metrics = metrics(x_pg, x_true);
    results.pg.metrics_best = metrics(info_pg.x_best, x_true);

    results.fista.x = x_fista;
    results.fista.info = info_fista;
    results.fista.metrics = metrics(x_fista, x_true);
    results.fista.metrics_best = metrics(info_fista.x_best, x_true);

    results.blurred_metrics = metrics(b, x_true);
end