import numpy as np
from PIL import Image

size = 1024
half_size = int(size / 2)
img = np.zeros([size, size, 3]).astype(np.uint8)

img[:, :] = np.array([0.2, 0.3, 0.4]) * 255
img[:half_size, :half_size] = np.array([0.1, 0.2, 0.3]) * 255
img[half_size:, half_size:] = np.array([0.1, 0.2, 0.3]) * 255

img[:, :, :] = 16
half_thickness = 1
img[:half_thickness] = 128
img[:, :half_thickness] = 128
img[-half_thickness:] = 128
img[:, -half_thickness:] = 128
img[:, half_size - half_thickness : half_size + half_thickness] = 128
img[half_size - half_thickness : half_size + half_thickness, :] = 128
Image.fromarray(img).save("checker_bw.png")
