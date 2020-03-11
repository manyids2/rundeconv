from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from viz import plt, image, save_figure

n = 50
paddings = [0, 1, 2, 3]
kernel_sizes = [3, 4, 5, 6, 7]
strides = [1, 2, 3, 4]

for p in paddings:
  for k in kernel_sizes:
    for s in strides:

      N = s * (n - 1) + k - (2 * p)

      deconv_fn = nn.ConvTranspose2d(in_channels=1,
                                     out_channels=1,
                                     kernel_size=k,
                                     stride=s,
                                     padding=p,
                                     bias=False)
      deconv_fn.weight[:] = 1.0

      ones_tensor = torch.ones((1, 1, n, n))
      output = deconv_fn(ones_tensor).detach().numpy()[0, 0, :, :]
      unique = np.unique(output)
      print(output.shape)

      plt.rcParams['figure.figsize'] = (30, 10)
      _, axs = plt.subplots(1, 3)

      def make_title(output):
        unique = np.unique(output)
        title = (
            f'n:{n}, s:{s}, p:{p}, k:{k}, N:{N}\n'
            f'{output.shape[-1]}x{output.shape[-1]}\n{len(unique)};{unique}')
        return title

      image(output,
            title=make_title(output),
            ax=axs[0],
            colorbar=True,
            ticks=True)

      # Copy
      stride_tile = np.zeros((s, s))
      stride_tile[0, 0] = 1.0
      big_ones = np.tile(stride_tile, (n, n))

      image(big_ones,
            title=make_title(big_ones),
            ax=axs[1],
            colorbar=True,
            ticks=True)

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

      image(conv_output,
            title=make_title(conv_output),
            ax=axs[2],
            colorbar=True,
            ticks=True)

      diff = np.sqrt(np.sum((output - conv_output)**2))
      print(f'diff:{diff}')
      plt.suptitle(f'diff:{diff}')

      savefile = Path(
          'data/deconv_ones') / f'n-{n}_s-{s}_p-{p}_k-{k}_N-{N}_diff-{diff}.png'
      save_figure(savefile)

      plt.close()
