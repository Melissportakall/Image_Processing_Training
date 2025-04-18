import cv2
import numpy as np
import matplotlib.pyplot as plt

# Gürültülü bir örnek görüntü oluştur (veya bir resim yükle)
img = cv2.imread('prokudin.jpg', cv2.IMREAD_GRAYSCALE)

# Gürültü eklemek istersen örnek (salt & pepper noise)
def add_salt_pepper_noise(image, amount=0.02):
    noisy = image.copy()
    total_pixels = image.size
    num_salt = int(amount * total_pixels / 2)
    num_pepper = int(amount * total_pixels / 2)

    # Salt
    coords = [np.random.randint(0, i, num_salt) for i in image.shape]
    noisy[coords[0], coords[1]] = 255

    # Pepper
    coords = [np.random.randint(0, i, num_pepper) for i in image.shape]
    noisy[coords[0], coords[1]] = 0

    return noisy

noisy_img = add_salt_pepper_noise(img)

# 3x3 medyan filtresi uygula
median_filtered = cv2.medianBlur(noisy_img, 3)

# Karşılaştırmalı görselleri göster
titles = ['Orijinal', 'Gürültülü', 'Median Filter']
images = [img, noisy_img, median_filtered]

plt.figure(figsize=(15, 5))

for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
