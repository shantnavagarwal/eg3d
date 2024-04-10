from PIL import Image
import numpy as np
from copy import copy

file_name = "out/test1"

# Densities
gif = Image.open(file_name + ".gif")

gif_arr = []
frame_indices = np.linspace(0, gif.n_frames - 1, 4, dtype=int)

for i, idx in enumerate(frame_indices):
    gif.seek(idx)
    frame = np.array(gif)
    frame = np.max(frame) - frame
    gif_arr.append(frame)

out = np.concatenate(gif_arr, axis=1)
out = Image.fromarray(out)
out.save(file_name + 'blog.png')
print("Done")

gif = Image.open(file_name + "c.gif")

gif_arr = []
frame_indices = np.linspace(0, gif.n_frames - 1, 4, dtype=int)

for i, idx in enumerate(frame_indices):
    gif.seek(idx)
    frame = np.array(gif)
    frame = np.max(frame) - frame
    gif_arr.append(frame)

out = np.concatenate(gif_arr, axis=1)
out = Image.fromarray(out)
out.save(file_name + 'blog_c.png')
print("Done")