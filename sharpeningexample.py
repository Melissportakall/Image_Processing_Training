import cv2
import numpy as np
import matplotlib.pyplot as plt

# Görüntüyü gri tonlamada yükle
img = cv2.imread('prokudin.jpg', cv2.IMREAD_GRAYSCALE)
# 1. Impulse kernel (ortası 2)
#parlaklık atıcak
impulse_kernel = np.zeros((3, 3), dtype=np.float32)
impulse_kernel[1, 1] = 2

# 2. Box filtre (ortalama blur)
box_kernel = np.ones((3, 3), dtype=np.float32) / 9

# 3. Sharpening filtresi: impulse - blur
sharpen_kernel = impulse_kernel - box_kernel

# 4. Filtrelerin uygulanması
impulse_result = cv2.filter2D(img, -1, impulse_kernel)
box_result     = cv2.filter2D(img, -1, box_kernel)
sharpen_result = cv2.filter2D(img, -1, sharpen_kernel)

# Tüm görüntüleri bir araya getir
titles = [
    "Orijinal", 
    "Impulse (ortası 2)", 
    "Box Blur", 
    "Sharpening (Impulse - Box)"
]
images = [img, impulse_result, box_result, sharpen_result]

plt.figure(figsize=(18, 5))

for i in range(len(images)):
    plt.subplot(1, 4, i + 1)
    plt.imshow(images[i], cmap='gray', vmin=0, vmax=255)
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
