import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = 'prokudin.jpg' 
image = cv2.imread(image_path)  

b_channel, g_channel, r_channel = cv2.split(image)

def align_channels(base, target):
    result = cv2.matchTemplate(target, base, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)
    
    dy, dx = max_loc
    return np.roll(target, shift=(dy, dx), axis=(0, 1))

aligned_g = align_channels(b_channel, g_channel)
aligned_r = align_channels(b_channel, r_channel)

combined_image = cv2.merge((b_channel, aligned_g, aligned_r))

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(b_channel, cmap='gray')
plt.title('B Kanalı')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(g_channel, cmap='gray')
plt.title('G Kanalı')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(r_channel, cmap='gray')
plt.title('R Kanalı')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)) 
plt.title('Hizalanmış Görüntü')
plt.axis('off')

plt.tight_layout()
plt.show()
