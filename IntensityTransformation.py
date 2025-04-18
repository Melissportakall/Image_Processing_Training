import cv2
import numpy as np
import matplotlib.pyplot as plt

# Gri tonlamalı görüntüyü yükle
img = cv2.imread("prokudin.jpg", cv2.IMREAD_GRAYSCALE)

# Görüntü gösterme fonksiyonu
def show(title, image):
    plt.figure()
    plt.title(title)
    plt.axis("off")
    plt.imshow(image, cmap="gray")
    plt.show()

# Normalize et (0-1 aralığına), sonra log uygula 
# s=c⋅log(1+r)
img_float = img / 255.0
log_transformed = np.log1p(img_float)  # log(1 + r)

# Normalle ve uint8'e dönüştür 

#s=10^(c⋅r)−1
log_transformed = cv2.normalize(log_transformed, None, 0, 255, cv2.NORM_MINMAX)
log_transformed = log_transformed.astype(np.uint8)

show("Log Transformation", log_transformed)

# normalize et ve ters log uygula
img_float = img / 255.0
inverse_log = np.exp(img_float) - 1

# normalle ve tip dönüşümü
inverse_log = cv2.normalize(inverse_log, None, 0, 255, cv2.NORM_MINMAX)
inverse_log = inverse_log.astype(np.uint8)

show("Inverse Log Transformation", inverse_log)

# normalize et ve gamma uygula
# s=c⋅r^γ
gamma = 2.0  # 0.5 aydınlatır,  >1 karartır
img_float = img / 255.0
gamma_corrected = np.power(img_float, gamma)

# normalle ve uint8'e dönüştür
gamma_corrected = (gamma_corrected * 255).astype(np.uint8)

show(f"Gamma Correction (gamma={gamma})", gamma_corrected)
