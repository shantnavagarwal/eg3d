import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from moviepy.editor import ImageSequenceClip
import cv2 as cv

def save_array(arr: np.ndarray, filename: str) -> None:
    np.save(filename, arr)
    return


def save_torch_array(arr: torch.Tensor, filename: str) -> None:
    save_array(arr.detach().cpu().numpy(), filename)
    return


def plot_densities_plots(densities_fine) -> None:
    if type(densities_fine) is torch.Tensor:
        densities_fine = densities_fine.detach().cpu().numpy()
    new_shape = int(np.sqrt(densities_fine.shape[1]))
    densities_fine = densities_fine[0, :, :, 0].reshape((new_shape, new_shape, densities_fine.shape[2]))
    densities_fine_stand = (densities_fine - np.min(densities_fine)) / (np.max(densities_fine) - np.min(densities_fine))
    densities_fine_last = np.sum(densities_fine_stand, axis=2).reshape((densities_fine_stand.shape[0], densities_fine_stand.shape[1], 1))
    densities_fine_last = (densities_fine_last - np.min(densities_fine_last)) / (np.max(densities_fine_last) - np.min(densities_fine_last))
    densities_fine_stand = np.append(densities_fine_stand, densities_fine_last, axis=2)

    frames = np.transpose(densities_fine_stand, (2, 0, 1))
    frames = np.expand_dims(frames, axis=3)*255
    clip = ImageSequenceClip(list(frames), fps=5)
    filename = 'out/test1'
    clip.write_gif(filename+'.gif', fps=20)
    cv.imwrite(filename+'.png', densities_fine_last*255)
    # plot_3d_array(densities_fine_stand)
    return


def plot_colors_plots(colors_fine) -> None:
    if type(colors_fine) is torch.Tensor:
        colors_fine = colors_fine.detach().cpu().numpy()
    new_shape = int(np.sqrt(colors_fine.shape[1]))
    colors_fine = colors_fine[:, :, :, :3]
    colors_fine = colors_fine[0, :, :, :].reshape((new_shape, new_shape, colors_fine.shape[2], colors_fine.shape[3]))
    colors_fine_stand = (colors_fine - np.min(colors_fine)) / (np.max(colors_fine) - np.min(colors_fine))
    colors_fine_last = np.sum(colors_fine_stand, axis=2).reshape((colors_fine_stand.shape[0], colors_fine_stand.shape[1], 1, colors_fine.shape[3]))
    colors_fine_last = (colors_fine_last - np.min(colors_fine_last)) / (np.max(colors_fine_last) - np.min(colors_fine_last))
    colors_fine_stand = np.append(colors_fine_stand, colors_fine_last, axis=2)

    frames = np.transpose(colors_fine_stand, (2, 0, 1, 3))
    frames = frames*255
    clip = ImageSequenceClip(list(frames), fps=5)
    filename = 'out/test1c'
    clip.write_gif(filename+'.gif', fps=20)
    cv.imwrite(filename+'.png', colors_fine_last[:, :, 0, :]*255)
    # plot_3d_array(densities_fine_stand)
    return

def plot_3d_array(array):
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    # Create slider
    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
    slider = Slider(ax_slider, 'Slice', 0, array.shape[2] - 1, valinit=0, valstep=1)

    # Initial plot
    img = ax.imshow(array[:, :, 0], cmap='viridis')

    def update(val):
        # Update plot when slider value changes
        slice_idx = int(slider.val)
        img.set_array(array[:, :, slice_idx])
        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()