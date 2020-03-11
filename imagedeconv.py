from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from viz import plt, image, read_image, save_figure

img = read_image('data/smug_girl_statue_1.jpg', flatten=True)
img = img[:128, :128]
img_tensor = torch.from_numpy(img[np.newaxis, np.newaxis, :, :].astype(
    np.float32))
ones_tensor = torch.ones((1, 1, 128, 128), dtype=torch.float32)

n = img.shape[0]
s = 5
k = 7
p = 5
N = s * (n - 1) + k - (2 * p)

deconv_fn = nn.ConvTranspose2d(in_channels=1,
                               out_channels=1,
                               kernel_size=k,
                               stride=s,
                               padding=p,
                               bias=False)
deconv_fn.weight[:] = 1.0
mask = deconv_fn(ones_tensor).detach().numpy()[0, 0, :, :]

deconv_fn = nn.ConvTranspose2d(in_channels=1,
                               out_channels=1,
                               kernel_size=k,
                               stride=s,
                               padding=p,
                               bias=False)
output = deconv_fn(img_tensor).detach().numpy()[0, 0, :, :]
unique = np.unique(output)
print(output.shape)

# Copy
stride_tile = np.zeros((s, s))
stride_tile[0, 0] = 1.0
big_ones = np.tile(stride_tile, (n, n))
h, w = img.shape
for i in range(h):
  for j in range(w):
    big_ones[s * i, s * j] = img[i, j]

big_ones_tensor = torch.from_numpy(
    big_ones[np.newaxis, np.newaxis, :, :].astype(np.float32))
conv_fn = nn.Conv2d(in_channels=1,
                    out_channels=1,
                    kernel_size=k,
                    stride=1,
                    padding=k - 1,
                    bias=False)
conv_fn.weight[:] = 1.0

N = s * (n - 1) + k - (2 * p)

conv_output = conv_fn(big_ones_tensor).detach().numpy()[0, 0, :, :]
conv_output = conv_output[p:N + p, p:N + p]
print(conv_output.shape)

plt.rcParams['figure.figsize'] = (30, 10)
_, axs = plt.subplots(1, 5)

image(img, title='image', ax=axs[0], colorbar=True, ticks=True)
image(mask, title='impulse', ax=axs[1], colorbar=True, ticks=True)
image(output, title='deconv', ax=axs[2], colorbar=True, ticks=True)
image(conv_output, title='conv', ax=axs[3], colorbar=True, ticks=True)
image(output / mask, title='demask', ax=axs[4], colorbar=True, ticks=True)

# savefile = Path('data/deconv_ones'
#                 ) / f'n-{n}_s-{s}_p-{p}_k-{k}_N-{N}_img1_random_kernel.png'
# save_figure(savefile)

plt.show()
plt.close()
