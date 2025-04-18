import cv2
import numpy as np
import matplotlib.pyplot as plt

#BU KAYMALAR CONVOLUTİON İŞLEMİNDE FİLTRENİN DÖNDÜRÜLMESİNDEN DOLAYI ORTAYA ÇIKAR.
# 5x5 basit test görüntüsü
#yatayda değişim için bunu kullanın
img = np.array([
    [  0,  50, 100, 150, 200],
    [  0,  50, 100, 150, 200],
    [  0,  50, 100, 150, 200],
    [  0,  50, 100, 150, 200],
    [  0,  50, 100, 150, 200]
], dtype=np.uint8)

# Dikeyde değişim için bunu kullanın

#img = np.array([
#    [  0,   0,   0,   0,   0],
#    [ 50,  50,  50,  50,  50],
#    [100, 100, 100, 100, 100],
#    [150, 150, 150, 150, 150],
#    [200, 200, 200, 200, 200]
#], dtype=np.uint8)


# Filtreler

# 1. Ortada 1 → değişim yok
identity_kernel = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
], dtype=np.float32)

# 2. Sağ → Görüntü sola kayar
right_kernel = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [0, 0, 0]
], dtype=np.float32)

# 3. Sol → Görüntü sağa kayar
left_kernel = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 0, 0]
], dtype=np.float32)

# 4. Aşağı → Görüntü yukarı kayar
down_kernel = np.array([
    [0, 0, 0],
    [0, 0, 0],
    [0, 1, 0]
], dtype=np.float32)

# 5. Yukarı → Görüntü aşağı kayar
up_kernel = np.array([
    [0, 1, 0],
    [0, 0, 0],
    [0, 0, 0]
], dtype=np.float32)

# Uygulamalar
out_identity = cv2.filter2D(img, -1, identity_kernel)
out_right    = cv2.filter2D(img, -1, right_kernel)
out_left     = cv2.filter2D(img, -1, left_kernel)
out_down     = cv2.filter2D(img, -1, down_kernel)
out_up       = cv2.filter2D(img, -1, up_kernel)

# Görselleri yan yana göster
titles = [
    'Original',
    'Ortada 1 (değişmez)',
    'Sağa kaymış 1 (çıktı sola)',
    'Sola kaymış 1 (çıktı sağa)',
    'Aşağı kaymış 1 (çıktı yukarı)',
    'Yukarı kaymış 1 (çıktı aşağı)'
]
images = [img, out_identity, out_right, out_left, out_down, out_up]

plt.figure(figsize=(18, 6))

for i in range(len(images)):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], cmap='gray', vmin=0, vmax=255)
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
