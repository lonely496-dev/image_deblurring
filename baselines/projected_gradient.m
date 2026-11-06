function [x, info] = projected_gradient(gradFun, b, psf, x0, opts)
% Plain projected gradient = SGP with D_k = I (no scaling).
% Same signature as sgp_solve, for direct comparison.

if nargin < 5
    opts = struct();
end
opts.use_scaling = false;
[x, info] = sgp_solve(gradFun, b, psf, x0, opts);
end