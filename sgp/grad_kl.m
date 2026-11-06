function [f, g, Vplus] = grad_kl(x, b, psf)
Hx = apply_blur(x, psf);
eps_v = 1e-10;
Hx_safe = max(Hx, eps_v);
term = Hx - b;
mask = b > 0;
term(mask) = term(mask) - b(mask) .* log(Hx_safe(mask) ./ b(mask));
f = sum(term(:));
if nargout > 1, g = apply_blur(1 - b ./ Hx_safe, psf, 'transpose'); end
if nargout > 2, Vplus = apply_blur(ones(size(x)), psf, 'transpose'); end
end