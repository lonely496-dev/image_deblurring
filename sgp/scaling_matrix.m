function D = scaling_matrix(x, Vplus, L, U)
% Split-gradient scaling: D = clip(x./Vplus, L, U)
% Vplus is the positive part of the split gradient ,

if nargin < 3 || isempty(L)
    L = 1e-5;
end
if nargin < 4 || isempty(U)
    U = 1e5;
end

eps_v = 1e-10;
D = x ./ max(Vplus, eps_v);
D = min(U, max(L, D));
end