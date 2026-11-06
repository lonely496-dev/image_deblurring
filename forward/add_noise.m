function b = add_noise(bx, noiseType, param)
% b = add_noise(Hx, 'gaussian', sigma)  -> Hx + N(0, sigma^2)
% b = add_noise(Hx, 'poisson', peak)    -> Poisson(peak*Hx)/peak
%
% peak controls Poisson SNR: higher peak = less noise (more photons).
% Requires Statistics and Machine Learning Toolbox (poissrnd).

switch lower(noiseType)
    case 'gaussian'
        if nargin < 3 || isempty(param)
            param = 0.01;
        end
        b = bx + param * randn(size(bx));
        b = max(b, 0);

    case 'poisson'
        if nargin < 3 || isempty(param)
            param = 100;
        end
        b = poissrnd(param * max(bx, 0)) / param;

    otherwise
        error('add_noise:badType', 'noiseType must be gaussian or poisson');
end
end