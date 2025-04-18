import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("prokudin.jpg",cv2.IMREAD_COLOR)

# image resize 
def resize(image, width, height):
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


plt.figure()
plt.title("show")
plt.axis("off")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

# Görüntüyü renkli yükle
img = cv2.imread("prokudin.jpg", cv2.IMREAD_COLOR)


# 100x100 boyutuna yeniden boyutlandır
resized_img = resize(img, 100, 100)


plt.figure()
plt.title("Resized Image")
plt.axis("off")
plt.imshow(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))
plt.show()
print("Resized image shape:", resized_img.shape)
print("Original image shape:", img.shape)