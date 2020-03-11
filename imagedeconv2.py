from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from viz import plt, image, read_image, save_figure

# Read image, resize to fit memory.
img = read_image('data/smug_girl_statue_1.jpg', flatten=True)
img = img[:128, :128]
img_tensor = torch.from_numpy(img[np.newaxis, np.newaxis, :, :].astype(
    np.float32))

# Tensor of ones.
ones_tensor = torch.ones((1, 1, 128, 128), dtype=torch.float32)

# Hyperparams - image-size, stride, kernel_size, padding, output-size
n = img.shape[0]
s = 5
k = 7
p = 5
N = s * (n - 1) + k - (2 * p)

# Deconvolution with kernel set to 1.0.
ones_deconv_fn = nn.ConvTranspose2d(in_channels=1,
                                    out_channels=1,
                                    kernel_size=k,
                                    stride=s,
                                    padding=p,
                                    bias=False)
ones_deconv_fn.weight[:] = 1.0

# Generate mask with kernel = 1.0.
mask_ones = ones_deconv_fn(ones_tensor).detach().numpy()[0, 0, :, :]

# Deconvolution with random kernel.
deconv_fn = nn.ConvTranspose2d(in_channels=1,
                               out_channels=1,
                               kernel_size=k,
                               stride=s,
                               padding=p,
                               bias=False)

# Kernel with 1.0.
kernel = deconv_fn.weight.detach().numpy()[0, 0, :, :]
ones_kernel = kernel.copy()
ones_kernel[:] = 1.0

# Generate mask with random kernel.
mask_wts = deconv_fn(ones_tensor).detach().numpy()[0, 0, :, :]
output = deconv_fn(img_tensor).detach().numpy()[0, 0, :, :]

# Plot everything.
plt.rcParams['figure.figsize'] = (30, 30)
_, axs = plt.subplots(3, 3)

image(img, title='image', ax=axs[0, 0], colorbar=True, ticks=True)
image(output, title='output', ax=axs[0, 1], colorbar=True, ticks=True)
image(kernel,
      title='kernel for deconv',
      ax=axs[0, 2],
      colorbar=True,
      ticks=True)

image(mask_ones, title='mask_ones', ax=axs[1, 0], colorbar=True, ticks=True)
image(output / mask_ones,
      title='output / mask_ones',
      ax=axs[1, 1],
      colorbar=True,
      ticks=True)
image(ones_kernel,
      title='kernel for mask',
      ax=axs[1, 2],
      colorbar=True,
      ticks=True)

image(mask_wts, title='mask_wts', ax=axs[2, 0], colorbar=True, ticks=True)
image(output / mask_wts,
      title='output / mask_wts',
      ax=axs[2, 1],
      colorbar=True,
      ticks=True)
image(kernel, title='kernel for mask', ax=axs[2, 2], colorbar=True, ticks=True)

# Save file.
savefile = Path('data/deconv_ones'
                ) / f'n-{n}_s-{s}_p-{p}_k-{k}_N-{N}_img1_random_kernel.png'
save_figure(savefile)

# Show plot.
plt.show()
plt.close()
