import cv2
import numpy as np
import matplotlib.pyplot as plt


#SOBEL FİLTRESİ:görüntüde parlaklık değişiminin yönünü ve büyüklüğünü bulur.
# Görüntüyü gri tonlamada yükle
img = cv2.imread('prokudin.jpg', cv2.IMREAD_GRAYSCALE)

# Sobel X ve Y filtrelerini uygula
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

# Kenar büyüklüğünü hesapla
sobel_mag = cv2.magnitude(sobel_x, sobel_y)

# Görselleri göster
plt.figure(figsize=(15, 5))
titles = ['Orijinal', 'Sobel X', 'Sobel Y', 'Kenarlar (Magnitude)']
images = [img, np.abs(sobel_x), np.abs(sobel_y), sobel_mag]

for i in range(4):
    plt.subplot(1, 4, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
