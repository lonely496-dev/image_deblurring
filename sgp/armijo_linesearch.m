function [lambda, x_new, f_new] = armijo_linesearch(fun, x, f_x, grad_x, dx, beta, delta, max_iter)
% fun: function handle, f = fun(x) (objective value only)
% Backtracks lambda from 1 until f(x+lambda*dx) <= f_x + beta*lambda*slope
if nargin < 6 || isempty(beta),     beta = 1e-4;   end
if nargin < 7 || isempty(delta),    delta = 0.5;   end
if nargin < 8 || isempty(max_iter), max_iter = 30; end

slope = sum(grad_x .* dx, 'all');
lambda = 1;
for m = 0:max_iter
    x_new = x + lambda * dx;
    f_new = fun(x_new);
    if f_new <= f_x + beta * lambda * slope
        return;
    end
    lambda = lambda * delta;
end
warning('armijo:noDecrease', 'Armijo failed after %d backtracks.', max_iter);
end