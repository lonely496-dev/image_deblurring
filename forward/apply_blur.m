function out = apply_blur(x, psf, mode)
% out = apply_blur(x, psf)              -> H*x
% out = apply_blur(x, psf, 'transpose') -> H'*x
% Circular convolution via FFT. x can be HxW or HxWxC.
if nargin < 3 || isempty(mode), mode = 'forward'; end
persistent keyPsf keySz Ffwd Ftr
sz = [size(x,1) size(x,2)];
if isempty(keyPsf) || ~isequal(keySz, sz) || ~isequal(keyPsf, psf)
    Ffwd = fft2(pad_kernel(psf, sz));
    Ftr  = fft2(pad_kernel(rot90(psf, 2), sz));
    keyPsf = psf; keySz = sz;
end
switch lower(mode)
    case 'forward',   F = Ffwd;
    case 'transpose', F = Ftr;
    otherwise, error('apply_blur:badMode', 'mode must be forward or transpose');
end
out = real(ifft2(fft2(x) .* F));
end

function p = pad_kernel(kernel, sz)
[kr, kc] = size(kernel);
p = zeros(sz);
p(1:kr, 1:kc) = kernel;
p = circshift(p, -floor([kr, kc] / 2));
end