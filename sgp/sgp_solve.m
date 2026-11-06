function [x, info] = sgp_solve(gradFun, b, psf, x0, opts)
% Scaled Gradient Projection.
% gradFun: [f,g,Vplus] = gradFun(x,b,psf)   
% x0: initial point 
% opts fields :
%   use_scaling  - true/false, default true (false = plain projected gradient)
%   bb_rule      - 'bb1','bb2','alternate', default 'alternate'
%   max_iter     - default 200
%   tol          - relative step tolerance, default 1e-5
%   l_bound/u_bound, alpha_min/max, D_L/D_U - bounds
%   x_true       

    if nargin < 5
        opts = struct();
    end
    use_scaling = getf(opts, 'use_scaling', true);
    bb_rule     = getf(opts, 'bb_rule', 'alternate');
    max_iter    = getf(opts, 'max_iter', 200);
    tol         = getf(opts, 'tol', 1e-5);
    l_bound     = getf(opts, 'l_bound', 0);
    u_bound     = getf(opts, 'u_bound', 1);
    alpha_min   = getf(opts, 'alpha_min', 1e-5);
    alpha_max   = getf(opts, 'alpha_max', 1e5);
    D_L         = getf(opts, 'D_L', 1e-5);
    D_U         = getf(opts, 'D_U', 1e5);
    x_true      = getf(opts, 'x_true', []);
    track_psnr  = ~isempty(x_true);

    x = x0;
    [f, g, Vplus] = gradFun(x, b, psf);

    info.f = zeros(max_iter, 1);
    info.alpha = zeros(max_iter, 1);
    info.gnorm = zeros(max_iter, 1);
    info.lambda = zeros(max_iter,1); info.D_low = zeros(max_iter,1); info.D_high = zeros(max_iter,1);
    if track_psnr
        info.psnr = zeros(max_iter, 1);
        best_psnr = -Inf; best_iter = 0; x_best = x0;
    end

    x_prev = [];
    g_prev = [];
    alpha = 1;

    for k = 1:max_iter
        if use_scaling
            D = scaling_matrix(x, Vplus, D_L, D_U);
            info.D_low(k)  = mean(D(:) <= D_L*1.0001);
            info.D_high(k) = mean(D(:) >= D_U*0.9999);
        else
            D = ones(size(x));
        end

        if ~isempty(x_prev)
            s = x - x_prev;
            z = g - g_prev;
            alpha = bb_steplength(s, z, D, bb_rule, alpha_min, alpha_max);
        end

        y = min(u_bound, max(l_bound, x - alpha * D .* g));
        dx = y - x;

        fun = @(xx) gradFun(xx, b, psf);
        [lambda, x_new, f_new] = armijo_linesearch(fun, x, f, g, dx);

        x_prev = x;
        g_prev = g;
        x = x_new;
        f = f_new;
        [~, g, Vplus] = gradFun(x, b, psf);

        info.f(k) = f;
        info.alpha(k) = alpha;
        info.gnorm(k) = norm(g(:));
        info.lambda(k) = lambda;

        if track_psnr
            info.psnr(k) = psnr(x, x_true);
            if info.psnr(k) > best_psnr
                best_psnr = info.psnr(k);
                best_iter = k;
                x_best = x;
            end
        end

        rel_step = norm(dx(:) * lambda) / max(norm(x(:)), 1e-10);
        if rel_step < tol
            break;
        end
    end

    info.f = info.f(1:k);
    info.alpha = info.alpha(1:k);
    info.gnorm = info.gnorm(1:k);
    info.lambda = info.lambda(1:k); info.D_low = info.D_low(1:k); info.D_high = info.D_high(1:k);
    info.iters = k;
    if track_psnr
        info.psnr = info.psnr(1:k);
        info.best_psnr = best_psnr;
        info.best_iter = best_iter;
        info.x_best = x_best;
    end
end

function v = getf(s, name, default)
    if isfield(s, name) && ~isempty(s.(name))
        v = s.(name);
    else
        v = default;
    end
end