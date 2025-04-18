import cv2
import numpy as np
import matplotlib.pyplot as plt

# Görüntüyü gri tonlamada yükle
img = cv2.imread('prokudin.jpg', cv2.IMREAD_GRAYSCALE)

# Hata kontrolü (önlem)
if img is None:
    print("❌ Görüntü yüklenemedi! Dosya adını kontrol et.")
    exit()

# 3x3 Box (ortalama) filtresi
box_filtered = cv2.blur(img, (3, 3))

# 3x3 Gaussian filtresi, sigmaX=1
gauss_filtered = cv2.GaussianBlur(img, (3, 3), sigmaX=1)

# Tent filtresi çekirdeği (üçgen ağırlıklı)
tent_kernel = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
], dtype=np.float32)
tent_kernel /= tent_kernel.sum()

# Tent filtresi uygulanıyor
tent_filtered = cv2.filter2D(img, -1, tent_kernel)

# Tüm filtre sonuçlarını yan yana göster
titles = ['Box Filter', 'Gaussian Filter', 'Tent Filter']
images = [box_filtered, gauss_filtered, tent_filtered]

plt.figure(figsize=(15, 5))

for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
