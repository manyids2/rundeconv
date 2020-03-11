import numpy as np
import colorcet as cc
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable
import ipywidgets as widgets
from IPython.display import display, Image
from PIL import Image


def read_image(filename, flatten=False, dtype=np.float32):
  img = Image.open(filename)
  img.load()
  data = np.asarray(img, dtype=dtype)
  if flatten:
    data = data.mean(axis=-1)
  return data


def optimal_grid(n):
  # TODO : do it properly
  ncols = int(np.sqrt(n))
  nrows = int(np.sqrt(n))
  return nrows, ncols


def inrange(x, e, s=0):
  x = np.array(x)
  x = (x - x.min()) / (x.max() - x.min())
  x = (x * (e - s)) - s
  return x


def figsize(h, w):
  plt.rcParams["figure.figsize"] = h, w


def imresize(img, size):
  img = np.array(Image.fromarray(img).resize(size))
  return img


def removeticks(ax):
  ax.axis('off')
  ax.set_xticks([])
  ax.set_yticks([])


def axifnone(ax, shape=[1], figsize=None):
  if ax is None:
    if figsize is None:
      figsize = (3, 3) if len(shape) == 1 else (shape[1] * 3, shape[0] * 3)
    _, ax = plt.subplots(*shape, figsize=figsize)
    return ax
  else:
    return ax


def save_figure(savefile, img=None):
  if savefile is not None:
    if img is None:
      plt.savefig(str(savefile),
                  dpi=300,
                  transparent=False,
                  bbox_inches='tight',
                  pad_inches=0.0,
                  metadata=None)
    else:
      plt.imsave(savefile, img)


def set_title(title, ax):
  if title is not None:
    ax.set_title(title)


def showpoints(points, ax, marker='+', color='r'):
  if points is not None:
    if not isinstance(color, list):
      color = [color] * len(points)
    ax.scatter(points[:, 0], points[:, 1], marker=marker, color=color)


def image(img,
          cmap=plt.cm.gray_r,
          title=None,
          ax=None,
          figsize=None,
          colorbar=False,
          ticks=False,
          bcolor=None,
          lw=5,
          saveplot=False,
          savefile=None,
          points=None):

  ax = axifnone(ax, figsize=figsize)
  im = ax.imshow(img.squeeze(), cmap=cmap, interpolation='nearest')
  showpoints(points, ax)

  if bcolor is not None:
    mark_box(ax, bcolor, lw)
  else:
    if not ticks:
      removeticks(ax)

  if colorbar:
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, ticks=[img.min(), img.max()])

  if title is not None:
    ax.set_title(title)

  if saveplot:
    save_figure(savefile)
  else:
    save_figure(savefile, img)


def browse_images(patches, titles=None, cmap=plt.cm.gray_r, colorbar=False):
  n = len(patches)

  def view_image(i):
    im = plt.imshow(patches[i], cmap=cmap, interpolation='nearest')
    if colorbar:
      divider = make_axes_locatable(plt.gca())
      cax = divider.append_axes("right", size="5%", pad=0.05)
      plt.colorbar(im, cax=cax)

  widgets.interact(view_image, i=(0, n - 1))


def layout_images(imgs,
                  shape=None,
                  titles=None,
                  wspace=-0.1,
                  hspace=0.1,
                  figsize=None,
                  cmap=plt.cm.gray_r,
                  ax=None,
                  savefile=None):
  n = len(imgs)
  if shape is None:
    shape = optimal_grid(n)
  ax = axifnone(ax, shape, figsize)
  plt.subplots_adjust(wspace=wspace, hspace=hspace)

  for i in range(shape[0]):
    for j in range(shape[1]):
      ax[i, j].imshow(imgs[i * shape[1] + j].squeeze(),
                      cmap=cmap,
                      interpolation='nearest')
      ax[i, j].axis('off')
      if titles is not None:
        set_title(titles[i * shape[1] + j], ax[i, j])

  save_figure(savefile)


def mark_box(ax, color, lw):
  plt.setp(ax.spines.values(), color=color, lw=lw)
  plt.setp([ax.get_xticklines(), ax.get_yticklines()], color=color)
  ax.set_xticks([])
  ax.set_yticks([])


def row_images(imgs,
               colorbar=False,
               column=False,
               titles=None,
               figsize=None,
               colors=None,
               hspace=None,
               wshapce=None,
               lw=5,
               cmap=plt.cm.gray_r,
               ax=None,
               savefile=None):
  n = len(imgs)
  ax = axifnone(ax, [n, 1] if column else [1, n], figsize)

  for i in range(n):
    im = ax[i].imshow(imgs[i].squeeze(), cmap=cmap, interpolation='nearest')
    if titles is not None:
      set_title(titles[i], ax[i])
    if colorbar:
      divider = make_axes_locatable(ax[i])
      cax = divider.append_axes("right", size="5%", pad=0.05)
      plt.colorbar(im, cax=cax)
    if colors is not None:
      mark_box(ax[i], colors[i], lw)
    else:
      removeticks(ax[i])

  plt.subplots_adjust(wspace=wshapce, hspace=hspace)
  save_figure(savefile)


def browse_list(patches_list, titles=None, cmaps=None, show_axis=False):
  M = len(patches_list)
  n = len(patches_list[0])

  if cmaps is None:
    cmaps = [cc.cm.fire] * M
  elif type(cmaps) is not list:
    cmaps = [cmaps] * M

  def view_pair(i):
    f, axarr = plt.subplots(1, M)
    for j in range(M):
      axarr[j].imshow(patches_list[j][i],
                      cmap=cmaps[j],
                      interpolation='nearest')
      if not show_axis:
        axarr[j].axis('off')
    if titles is None:
      f.suptitle(f'patch {i}')
    else:
      f.suptitle(titles[i])

  widgets.interact(view_pair, i=(0, n - 1))


def view_matches(img1, img2, xyB, xyA, figsize=(10, 10), savefile=None):
  ax1, ax2 = axifnone(None, [1, 2], figsize)
  ax1.imshow(img1)
  ax2.imshow(img2)
  removeticks(ax1)
  removeticks(ax2)
  assert xyB.shape == xyA.shape, 'check shapes'
  n = xyA.shape[0]
  for i in range(n):
    con = ConnectionPatch(xyA=xyA[i],
                          xyB=xyB[i],
                          coordsA="data",
                          coordsB="data",
                          axesA=ax2,
                          axesB=ax1,
                          color='red',
                          alpha=0.5)
    ax2.add_artist(con)
  save_figure(savefile)


def LAF2pts(LAF, n_pts=50):
  a = np.linspace(0, 2 * np.pi, n_pts)
  x = [0]
  x.extend(list(np.sin(a)))
  x = np.array(x).reshape(1, -1)
  y = [0]
  y.extend(list(np.cos(a)))
  y = np.array(y).reshape(1, -1)
  HLAF = np.concatenate([LAF, np.array([0, 0, 1]).reshape(1, 3)])
  H_pts = np.concatenate([x, y, np.ones(x.shape)])
  H_pts_out = np.transpose(np.matmul(HLAF, H_pts))
  H_pts_out[:, 0] = H_pts_out[:, 0] / H_pts_out[:, 2]
  H_pts_out[:, 1] = H_pts_out[:, 1] / H_pts_out[:, 2]
  return H_pts_out[:, 0:2]


def visualize_LAFs(img,
                   LAFs,
                   color='r',
                   pcolor='b',
                   marker='+',
                   lw=3,
                   ax=None,
                   points=None,
                   colors=None,
                   show=False,
                   savefile=None,
                   figsize=(20, 20)):
  ax = axifnone(ax, figsize=figsize)

  ax.imshow(img)
  removeticks(ax)
  showpoints(points, ax, color=pcolor, marker=marker)

  for i in range(len(LAFs)):
    ell = LAF2pts(LAFs[i, :, :])
    c = color if colors is None else colors[i]
    plt.plot(ell[:, 0], ell[:, 1], color=c, lw=lw)

  save_figure(savefile)

  return ax


def colorsandpatches(patches,
                     colors,
                     ax=None,
                     lw=5,
                     cmap=cc.cm.gray,
                     figsize=None,
                     savefile=None):
  n = len(patches)
  axs = axifnone(ax, shape=[n, 1], figsize=figsize)

  for i, ax in enumerate(axs):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(patches[i], cmap=cmap)
    for sp in ['bottom', 'top', 'left', 'right']:
      ax.spines[sp].set_color(colors[i])
      ax.spines[sp].set_linewidth(lw)

  plt.subplots_adjust(wspace=0, hspace=0.03)
  save_figure(savefile)
