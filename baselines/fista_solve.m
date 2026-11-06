function [x, info] = fista_solve(gradFun, b, psf, x0, opts)
% Monotone FISTA (MFISTA, Beck & Teboulle 2009): accelerated projected

if nargin < 5
    opts = struct();
end
max_iter = getf(opts, 'max_iter', 200);
tol      = getf(opts, 'tol', 1e-5);
l_bound  = getf(opts, 'l_bound', 0);
u_bound  = getf(opts, 'u_bound', 1);
L        = getf(opts, 'L0', 1);
eta      = getf(opts, 'eta', 2);
x_true   = getf(opts, 'x_true', []);
track_psnr = ~isempty(x_true);

x = x0;
y = x0;
t = 1;
f_x = gradFun(x, b, psf);

info.f = zeros(max_iter, 1);
info.L = zeros(max_iter, 1);
if track_psnr
    info.psnr = zeros(max_iter, 1);
    best_psnr = -Inf; best_iter = 0; x_best = x0;
end

for k = 1:max_iter
    [fy, gy] = gradFun(y, b, psf);

    for i = 0:50
        Lbar = L * eta^i;
        z = min(u_bound, max(l_bound, y - (1/Lbar) * gy));
        d = z - y;
        fz = gradFun(z, b, psf);
        rhs = fy + sum(gy(:) .* d(:)) + (Lbar/2) * sum(d(:).^2);
        if fz <= rhs + 1e-12
            break;
        end
    end
    L = Lbar;

    if fz <= f_x
        x_new = z;
        f_new = fz;
    else
        x_new = x;
        f_new = f_x;
    end

    t_next = (1 + sqrt(1 + 4*t^2)) / 2;
    y = x_new + (t / t_next) * (z - x_new) + ((t - 1) / t_next) * (x_new - x);

    step = x_new - x;
    x = x_new;
    f_x = f_new;
    t = t_next;

    info.f(k) = f_x;
    info.L(k) = L;

    if track_psnr
        info.psnr(k) = psnr(x, x_true);
        if info.psnr(k) > best_psnr
            best_psnr = info.psnr(k);
            best_iter = k;
            x_best = x;
        end
    end

    rel_step = norm(step(:)) / max(norm(x(:)), 1e-10);
    if rel_step < tol
        break;
    end
end

info.f = info.f(1:k);
info.L = info.L(1:k);
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