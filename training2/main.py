import cv2
import matplotlib.pyplot as plt
import numpy as np
#220201035 Melis Portakal

image = cv2.imread('kopek.jpg')

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imsave('kopek_saved_with_plt.jpg', image_rgb)

plt.imshow(image_rgb)
plt.axis('off')
plt.show()

print("Görüntü Numpy dizisi:\n", image_rgb )
print("Görüntü şekil:\n", image_rgb.shape)
print("Görüntü veri tipi:\n", image_rgb.dtype)
print("piksel değer aralığı:\n", image_rgb.min(), image_rgb.max())

b, g, r = cv2.split(image)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].imshow(b, cmap='gray')  
axs[0].set_title('Blue Kanalı (Gri Tonlamalı)')
axs[0].axis('off')

axs[1].imshow(g, cmap='gray')  
axs[1].set_title('Green Kanalı (Gri Tonlamalı)')
axs[1].axis('off')

axs[2].imshow(r, cmap='gray')  
axs[2].set_title('Red Kanalı (Gri Tonlamalı)')
axs[2].axis('off')

plt.tight_layout()
plt.show()

added = cv2.add(image, 50)   #add every piksel 50 and increase brightness   
subtracted = cv2.subtract(image, 50)  #subtract every piksel 50 and decrease brightness
multiplied = cv2.multiply(image, 1.5) #multiplact every piksel 1.5 and increase contrast
divided = cv2.divide(image, 2)  #divide every piksel 2 and make pale     

fig, axs = plt.subplots(1, 4, figsize=(20, 5))

axs[0].imshow(cv2.cvtColor(added, cv2.COLOR_BGR2RGB))
axs[0].set_title('Toplama (+50)')
axs[0].axis('off')

axs[1].imshow(cv2.cvtColor(subtracted, cv2.COLOR_BGR2RGB))
axs[1].set_title('Çıkarma (-50)')
axs[1].axis('off')

axs[2].imshow(cv2.cvtColor(multiplied.astype(np.uint8), cv2.COLOR_BGR2RGB))
axs[2].set_title('Çarpma (×1.5)')
axs[2].axis('off')

axs[3].imshow(cv2.cvtColor(divided.astype(np.uint8), cv2.COLOR_BGR2RGB))
axs[3].set_title('Bölme (÷2)')
axs[3].axis('off')

plt.tight_layout()
plt.show()

colors = ('b', 'g', 'r')
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

for i, color in enumerate(colors):
    hist = cv2.calcHist([image], [i], None, [256], [0, 256])
    axs[i].plot(hist, color=color)
    axs[i].set_title(f'{color.upper()} Kanalı Histogramı')
    axs[i].set_xlim([0, 256])

plt.tight_layout()
plt.show()