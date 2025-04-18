import cv2
import numpy as np
import matplotlib.pyplot as plt

# 7x7 boyutlu sıfır matrisi ve ortasına 1 koyarak impulse matrisi oluştur
impulse = np.zeros((7, 7), dtype=np.float32)
impulse[3, 3] = 1  # tam ortası

# BOX filtre (3x3 ortalama)
box_kernel = np.ones((3, 3), dtype=np.float32) / 9
box_output = cv2.filter2D(impulse, -1, box_kernel)

# GAUSSIAN benzeri filtre (3x3)
gauss_kernel = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
], dtype=np.float32)
gauss_kernel /= gauss_kernel.sum()
gauss_output = cv2.filter2D(impulse, -1, gauss_kernel)

# TENT filtresi (aynı Gaussian'a benziyor ama teorik olarak üçgen)
tent_kernel = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
], dtype=np.float32)
tent_kernel /= tent_kernel.sum()
tent_output = cv2.filter2D(impulse, -1, tent_kernel)

# Görselleri yan yana göster
titles = ['BOX Filter', 'Gaussian Filter', 'Tent Filter']
images = [box_output, gauss_output, tent_output]

plt.figure(figsize=(15, 5))

for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.imshow(images[i], cmap='gray', vmin=0, vmax=1)  # normalize görünüm
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
