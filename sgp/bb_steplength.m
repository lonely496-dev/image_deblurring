function alpha = bb_steplength(s, z, D, rule, alpha_min, alpha_max)
% BB steplength in the D-scaled metric (Bonettini et al. 2009).
%   BB1 = (s'*D^-1*D^-1*s) / (s'*D^-1*z)
%   BB2 = (s'*D*z)         / (z'*D*D*z)
% Nonpositive curvature estimate -> alpha_max (standard BB safeguard).
if nargin < 4 || isempty(rule),      rule = 'alternate'; end
if nargin < 5 || isempty(alpha_min), alpha_min = 1e-5;   end
if nargin < 6 || isempty(alpha_max), alpha_max = 1e5;    end

eps_d = 1e-12;
D2 = D.^2;
sDinv2_s = sum((s.^2) ./ D2, 'all');
sDinv_z  = sum((s .* z) ./ D, 'all');
sDz      = sum(s .* D .* z, 'all');
zD2z     = sum(D2 .* (z.^2), 'all');

if sDinv_z > 0
    alpha_bb1 = sDinv2_s / sDinv_z;
else
    alpha_bb1 = alpha_max;
end
if sDz > 0 && zD2z > 0
    alpha_bb2 = sDz / zD2z;
else
    alpha_bb2 = alpha_max;
end

switch lower(rule)
    case 'bb1', alpha = alpha_bb1;
    case 'bb2', alpha = alpha_bb2;
    case 'alternate'
        tau = 0.5;
        if alpha_bb2 / max(alpha_bb1, eps_d) < tau
            alpha = alpha_bb2;
        else
            alpha = alpha_bb1;
        end
    otherwise
        error('bb_steplength:badRule', 'rule must be bb1, bb2, or alternate');
end

if ~isfinite(alpha) || alpha <= 0
    alpha = alpha_min;
end
alpha = min(alpha_max, max(alpha_min, alpha));
end