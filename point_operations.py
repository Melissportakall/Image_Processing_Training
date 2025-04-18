import cv2
import numpy as np
import matplotlib.pyplot as plt

# Görüntüyü yükle (grayscale modda)
img = cv2.imread("prokudin.jpg", cv2.IMREAD_GRAYSCALE)

# Görüntü gösterme fonksiyonu
def show(title, image):
    plt.figure()
    plt.title(title)
    plt.axis('off')
    plt.imshow(image, cmap='gray')
    plt.show()  # Görüntüyü gerçekten göster

# Fonksiyonu çağır
show("Identity (Orijinal Görüntü)", img)

negative = 255 - img
show("Negative Image", negative)

# Parlaklaştır
bright = cv2.add(img, 50)
show("Brightened Image", bright)

# Karart
dark = cv2.subtract(img, 50)
show("Darkened Image", dark)

#Eşikleme
# Piksel değeri 128'in üzerindeyse 255, değilse 0 yap
_, thresholded = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
show("Thresholded Image", thresholded)

min_val = np.min(img)
max_val = np.max(img)


#Contrast stretching (Kontrast genişletme)
# min-max normalizasyonu
contrast_stretched = ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
show("Contrast Stretched", contrast_stretched)

#Intensity-Level Slicing (Yoğunluk Dilimleme)
# 100–150 aralığındaki pikselleri 255 yap, diğerlerini 0
sliced = np.zeros_like(img)
sliced[(img >= 100) & (img <= 150)] = 255
show("Intensity-Level Slicing", sliced)

#histogram gösterimi
plt.figure()
plt.title("Histogram")
plt.hist(img.ravel(), bins=256, range=[0, 256])
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.show()

