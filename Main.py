import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.util import random_noise
import os

# ============================================
# 1. LOAD IMAGE
# ============================================
image_path = 'sample_image.jpg'

if not os.path.exists(image_path):
    print(f"❌ Image '{image_path}' not found! Creating test image...")
    test_image = np.ones((400, 400, 3), dtype=np.uint8) * 200
    test_image[100:300, 100:300] = [0, 0, 255]
    test_image[150:250, 150:250] = [0, 255, 0]
    image_rgb = test_image
else:
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (400, 400))

# ============================================
# 2. ADD NOISE
# ============================================
gaussian_noise = random_noise(image_rgb, mode='gaussian', mean=0, var=0.002)
sp_noise = random_noise(image_rgb, mode='s&p', amount=0.02)

# ============================================
# 3. APPLY FILTERS
# ============================================
def apply_filter(img_float, filter_func, *args):
    img_uint8 = (img_float * 255).astype(np.uint8)
    filtered = filter_func(img_uint8, *args)
    return filtered.astype(np.float64) / 255.0

# Gaussian Filter
gaussian_on_gaussian = apply_filter(gaussian_noise, cv2.GaussianBlur, (5, 5), 1.5)
gaussian_on_sp = apply_filter(sp_noise, cv2.GaussianBlur, (5, 5), 1.5)

# NLM Filter
nlm_on_gaussian = apply_filter(gaussian_noise, cv2.fastNlMeansDenoisingColored, None, 10, 10, 7, 21)
nlm_on_sp = apply_filter(sp_noise, cv2.fastNlMeansDenoisingColored, None, 10, 10, 7, 21)

# Median Filter
median_on_gaussian = apply_filter(gaussian_noise, cv2.medianBlur, 5)
median_on_sp = apply_filter(sp_noise, cv2.medianBlur, 5)

# ============================================
# 4. CALCULATE PSNR
# ============================================
original = image_rgb.astype(np.float64) / 255.0

psnr_gauss_g = psnr(original, gaussian_on_gaussian)
psnr_gauss_sp = psnr(original, gaussian_on_sp)

psnr_nlm_g = psnr(original, nlm_on_gaussian)
psnr_nlm_sp = psnr(original, nlm_on_sp)

psnr_median_g = psnr(original, median_on_gaussian)
psnr_median_sp = psnr(original, median_on_sp)

# ============================================
# 5. DISPLAY RESULTS
# ============================================
fig = plt.figure(figsize=(14, 18))
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.1, hspace=0.25)

# Row 1: Original Image
ax1 = plt.subplot(5, 2, (1, 2))
ax1.imshow(image_rgb)
ax1.set_title('Original Image', fontsize=16)
ax1.axis('off')

# Row 2: Noisy Images
ax2 = plt.subplot(5, 2, 3)
ax2.imshow(gaussian_noise)
ax2.set_title('Gaussian Noise', fontsize=12)
ax2.axis('off')

ax3 = plt.subplot(5, 2, 4)
ax3.imshow(sp_noise)
ax3.set_title('Salt & Pepper Noise', fontsize=12)
ax3.axis('off')

# Row 3: Gaussian Filter
ax4 = plt.subplot(5, 2, 5)
ax4.imshow(gaussian_on_gaussian)
ax4.set_title(f'Gaussian on Gaussian\nPSNR: {psnr_gauss_g:.1f} dB', fontsize=10)
ax4.axis('off')

ax5 = plt.subplot(5, 2, 6)
ax5.imshow(gaussian_on_sp)
ax5.set_title(f'Gaussian on S&P\nPSNR: {psnr_gauss_sp:.1f} dB', fontsize=10)
ax5.axis('off')

# Row 4: NLM Filter
ax6 = plt.subplot(5, 2, 7)
ax6.imshow(nlm_on_gaussian)
ax6.set_title(f'NLM on Gaussian\nPSNR: {psnr_nlm_g:.1f} dB', fontsize=10)
ax6.axis('off')

ax7 = plt.subplot(5, 2, 8)
ax7.imshow(nlm_on_sp)
ax7.set_title(f'NLM on S&P\nPSNR: {psnr_nlm_sp:.1f} dB', fontsize=10)
ax7.axis('off')

# Row 5: Median Filter
ax8 = plt.subplot(5, 2, 9)
ax8.imshow(median_on_gaussian)
ax8.set_title(f'Median on Gaussian\nPSNR: {psnr_median_g:.1f} dB', fontsize=10)
ax8.axis('off')

ax9 = plt.subplot(5, 2, 10)
ax9.imshow(median_on_sp)
ax9.set_title(f'Median on S&P\nPSNR: {psnr_median_sp:.1f} dB', fontsize=10)
ax9.axis('off')

plt.tight_layout()
plt.show()

# ============================================
# 6. PRINT SUMMARY (PSNR Comparison)
# ============================================
print("\n" + "=" * 50)
print("PSNR Comparison (Higher is better)")
print("=" * 50)
print(f"Gaussian Filter:  Gaussian: {psnr_gauss_g:.1f} dB,  S&P: {psnr_gauss_sp:.1f} dB")
print(f"NLM Filter:       Gaussian: {psnr_nlm_g:.1f} dB,  S&P: {psnr_nlm_sp:.1f} dB")
print(f"Median Filter:    Gaussian: {psnr_median_g:.1f} dB,  S&P: {psnr_median_sp:.1f} dB")
print("\n✅ Done!")
