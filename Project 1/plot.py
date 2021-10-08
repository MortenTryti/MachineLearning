import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
# Load the terrain
terrain1 = imread("SRTM_data_Norway_1.tif")
# Show the terrain
print(type(terrain1))
print(terrain1.shape)
plt.figure()
plt.title("Terrain over Norway 1")
plt.imshow(terrain1)
plt.xlabel("’X’")
plt.colorbar()
plt.ylabel("’Y’")
plt.show()
