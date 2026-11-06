function m = metrics(x_est, x_true)
% m.psnr, m.ssim, m.relerr comparing x_est against ground truth x_true
m.psnr = psnr(x_est, x_true);
m.ssim = ssim(x_est, x_true);
m.relerr = norm(x_est(:) - x_true(:)) / norm(x_true(:));
end