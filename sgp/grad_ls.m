function [f, g, Vplus] = grad_ls(x, b, psf)
Hx = apply_blur(x, psf);
r = Hx - b;
f = 0.5 * sum(r(:).^2);
if nargout > 1, g = apply_blur(r, psf, 'transpose'); end
if nargout > 2, Vplus = apply_blur(Hx, psf, 'transpose'); end
end