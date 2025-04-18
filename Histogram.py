import cv2
import numpy as np
import matplotlib.pyplot as plt

# Görüntüyü yükle (grayscale)
img = cv2.imread("prokudin.jpg", cv2.IMREAD_GRAYSCALE)

# Görüntü ve histogramı göster
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.title("Orijinal Görüntü")
plt.axis("off")
plt.imshow(img, cmap='gray')

plt.subplot(1, 2, 2)
plt.title("Histogram")
plt.hist(img.ravel(), bins=256, range=[0,256], color='gray')
plt.xlabel("Piksel Değeri")
plt.ylabel("Frekans")

plt.tight_layout()
plt.show()


# Parlaklık artır
bright = cv2.add(img, 50)

# Kontrast artır (örnek olarak multiply yöntemiyle)
contrast = cv2.convertScaleAbs(img, alpha=1.5, beta=0)

# Hepsini karşılaştıralım
fig, axs = plt.subplots(3, 2, figsize=(10, 8))

axs[0, 0].imshow(img, cmap='gray')
axs[0, 0].set_title("Orijinal")
axs[0, 1].hist(img.ravel(), bins=256, range=[0,256])

axs[1, 0].imshow(bright, cmap='gray')
axs[1, 0].set_title("Parlaklık Artırılmış")
axs[1, 1].hist(bright.ravel(), bins=256, range=[0,256])

axs[2, 0].imshow(contrast, cmap='gray')
axs[2, 0].set_title("Kontrast Artırılmış")
axs[2, 1].hist(contrast.ravel(), bins=256, range=[0,256])

for ax in axs.flat:
    ax.axis("off") if ax in axs[:, 0] else None

plt.tight_layout()
plt.show()

#Nedir Histogram Equalization?
#Görüntüdeki parlaklık değerlerini (intensity) daha eşit dağıtarak kontrastı artırır.

#Özellikle düşük kontrastlı, gri ve sisli görüntülerde kullanılır.

#Amaç: histogramı olabildiğince düzgün (uniform) hale getirmek.

# Histogram eşitleme
equalized = cv2.equalizeHist(img)

# Görselleri ve histogramları karşılaştıralım
fig, axs = plt.subplots(2, 2, figsize=(10, 6))

# Orijinal görüntü
axs[0, 0].imshow(img, cmap='gray')
axs[0, 0].set_title("Orijinal Görüntü")
axs[0, 0].axis("off")
axs[1, 0].hist(img.ravel(), bins=256, range=[0, 256], color='gray')
axs[1, 0].set_title("Orijinal Histogram")

# Eşitlenmiş görüntü
axs[0, 1].imshow(equalized, cmap='gray')
axs[0, 1].set_title("Histogram Eşitleme Sonucu")
axs[0, 1].axis("off")
axs[1, 1].hist(equalized.ravel(), bins=256, range=[0, 256], color='gray')
axs[1, 1].set_title("Eşitlenmiş Histogram")

plt.tight_layout()
plt.show()