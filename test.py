import numpy as np
import matplotlib.pyplot as plt
import cv2
import pyDHM
from pyDHM import numericalPropagation
def pyqpi_reconstruct(I, wavelength, dx, z):
    F = np.fft.fftshift(np.fft.fft2(I))
    mag = np.abs(F)

    h, w = I.shape
    cy, cx = h//2, w//2

    # 去0级
    r0 = int(min(h, w)*0.1)
    mag[cy-r0:cy+r0, cx-r0:cx+r0] = 0

    # 找 +1级
    py, px = np.unravel_index(np.argmax(mag), mag.shape)

    # 滤波
    r = int(min(h, w)*0.06)
    Y, X = np.ogrid[:h, :w]
    mask = ((X-px)**2 + (Y-py)**2) < r**2
    F_filtered = F * mask

    # 搬移
    F_shifted = np.roll(F_filtered, cy-py, axis=0)
    F_shifted = np.roll(F_shifted, cx-px, axis=1)

    # 逆FFT
    field = np.fft.ifft2(np.fft.ifftshift(F_shifted))
    # 反传播
    field = numericalPropagation.angularSpectrum(
        field, -z, wavelength, dx, dx
    )
    return np.angle(field)
wavelength = 632.8e-9   #波长
A = 1.0     #入射光振幅
phi = 0.0   #入射光相位
z = 1e-3   #传播距离
dx = 1e-6   #采样间距
sample_light = 0.5    #假设部分透光
sample_phase_change =  0.5 * np.pi  #假设通过后的相位调制
theta = 60 * np.pi / 180 #参考光倾斜角度
I = cv2.imread('hologram.jpg', cv2.IMREAD_GRAYSCALE)
I2 = pyqpi_reconstruct(I,wavelength,dx,z)
plt.imshow(I2, cmap='gray')
plt.title("recovered")
plt.show()
cv2.imwrite('recover.jpg', (I2*255).astype(np.uint8))