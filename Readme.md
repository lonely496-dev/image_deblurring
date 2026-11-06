# Scaled Gradient Projection for Constrained Image Deblurring

Implementation compares three first-order methods for constrained image deblurring: Scaled Gradient Projection (SGP), plain Projected Gradient (PG) and monotone FISTA (MFISTA). They are tested on BSD500 images under Gaussian noise (least-squares objective) and Poisson noise (KL objective).

## Methods

- **SGP** follows Bonettini, Zanella and Zanni (2009): split-gradient diagonal scaling, scaled Barzilai-Borwein steplength and an Armijo line search.
- **PG** is SGP with the scaling matrix set to the identity. Everything else is shared, so it isolates the effect of scaling.
- **FISTA** is projected gradient with Nesterov momentum, using the monotone variant with backtracking Lipschitz estimation.

The degradation model is `b = Hx + n`, where `H` is circular convolution with a point-spread function (periodic boundaries). The solution is constrained to `[0, 1]` and there is no regularization term.

## Results

30 BSD500 training images, Gaussian blur (7x7, sigma 1.5), Gaussian noise (std 0.02), 80 iterations:

| Method | Mean best PSNR (dB) | Mean best iteration |
| ------ | ------------------- | ------------------- |
| SGP    | 23.5038             | 25.27               |
| PG     | 23.4574             | 24.77               |
| FISTA  | 23.4592             | 24.50               |

- Peak quality is nearly the same for all three. SGP is ahead by under 0.05 dB.
- Every method shows semiconvergence: PSNR peaks within the first ~20 iterations, then drops while the objective keeps decreasing. A lower objective does not mean a better image.
- SGP scaling helps on the KL objective (lower objective than PG throughout) but makes almost no difference on least squares.

"Best" PSNR is measured against the ground truth, so it is an oracle stopping rule and not something you could use on real data.

## Requirements

- MATLAB R2026b
- Image Processing Toolbox
- Statistics and Machine Learning Toolbox (for `poissrnd`)

## Setup

```bash
git clone https://github.com/lonely496-dev/image_deblurring.git
cd image_deblurring
```

```matlab
addpath('src');
```

The BSD500 dataset is downloaded automatically into `data/` the first time it is needed (about 70 MB).

## Usage

Run all three methods on one image:

```matlab
[imgs, names] = load_bsd500(fullfile('data', 'BSDS500-master'), 'train');
x_true = rgb2gray(imgs{1});

psf  = build_psf('gaussian', 7, 1.5);
opts = struct('max_iter', 300, 'tol', 0);

results = run_experiment(x_true, psf, 'gaussian', 0.02, opts);
```

Evaluate over many images and save the summary figures:

```matlab
opts    = struct('max_iter', 80, 'tol', 0);
summary = batch_eval(30, psf, 'gaussian', 0.02, opts);
make_figures(summary, 'results');
```

For Poisson noise, use `'poisson'` with a peak value (for example `60`) in place of the Gaussian arguments.

All solvers have the same signature, `[x, info] = solver(gradFun, b, psf, x0, opts)`, where `gradFun` is `@grad_ls` (Gaussian noise) or `@grad_kl` (Poisson noise). Set `opts.x_true` to track PSNR per iteration and get the best iterate in `info.x_best`.

## Repository structure

```
src/
  sgp_solve.m            Scaled Gradient Projection
  projected_gradient.m   PG baseline
  fista_solve.m          monotone FISTA
  grad_ls.m, grad_kl.m   objectives and gradients
  scaling_matrix.m, bb_steplength.m, armijo_linesearch.m
  build_psf.m, apply_blur.m, add_noise.m, load_bsd500.m
  metrics.m              PSNR, SSIM, relative error
  run_experiment.m       one image, all three methods
  batch_eval.m           many images
  make_figures.m         summary figures
data/                    BSD500 (downloaded on first use)
results/                 saved figures
```

Keep `batch_eval.m` inside `src/`, since it finds the repository root from its own location.

## References

1. S. Bonettini, R. Zanella, L. Zanni. A scaled gradient projection method for constrained image deblurring. _Inverse Problems_, 25(1), 2009.
2. A. Beck, M. Teboulle. A fast iterative shrinkage-thresholding algorithm for linear inverse problems. _SIAM Journal on Imaging Sciences_, 2(1), 2009.
3. J. Barzilai, J. M. Borwein. Two-point step size gradient methods. _IMA Journal of Numerical Analysis_, 8(1), 1988.
4. D. Martin, C. Fowlkes, D. Tal, J. Malik. A database of human segmented natural images and its application to evaluating segmentation algorithms and measuring ecological statistics. _ICCV_, 2001.

## License

MIT. See [LICENSE](LICENSE).
