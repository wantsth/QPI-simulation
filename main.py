import numpy as np
import matplotlib.pyplot as plt
import cv2
import pyDHM
from pyDHM import numericalPropagation
def image_process(name):
    image = cv2.imread(name, cv2.IMREAD_GRAYSCALE)    #读取图像
    matrix = image/255.0  #转化为矩阵
    return matrix   #有厚度的样本输出

def Transmission(U_in, matrix, wavelength, sample_light, sample_phase_change):
    # 投射函数
    A = 1 - (1 - sample_light) * matrix
    phi = sample_phase_change * matrix
    T = A * np.exp(1j * phi)

    # 出射样本光
    U_out = U_in * T
    return U_out

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

    # ⭐ 反传播
    from pyDHM import numericalPropagation
    field = numericalPropagation.angularSpectrum(
        field, -z, wavelength, dx, dx
    )

    return np.angle(field)

if __name__ == "__main__":
    wavelength = 632.8e-9   #波长
    A = 1.0     #入射光振幅
    phi = 0.0   #入射光相位
    z = 1e-3   #传播距离
    dx = 1e-6   #采样间距
    sample_light = 0.5    #假设部分透光
    sample_phase_change =  0.5 * np.pi  #假设通过后的相位调制
    theta = 10 * np.pi / 180 #参考光倾斜角度

    matrix = image_process('cameraman.jpg')   #图像处理
    print("matrix shape:", matrix.shape)

    h, w = matrix.shape
    U_in = np.ones((h, w))* A* np.exp(1j * 0)   #空间域光,初始光
    print("U_in shape:", U_in.shape)

    kx = (2*np.pi / wavelength) * np.sin(theta)
    x = np.arange(w)
    y = np.arange(h)
    X, Y = np.meshgrid(x, y)
    U_ref = np.exp(1j * kx * X * dx)

    sample_beam = Transmission(U_in, matrix, wavelength, sample_light, sample_phase_change)    #出射的频域光
    prop_field = numericalPropagation.angularSpectrum(sample_beam, z, wavelength, dx, dx)

    print("U_out shape:", prop_field)
    I = np.abs(U_ref + prop_field)**2
    print("I shape:", I.shape)
    cv2.imwrite('hologram.jpg', I)
    plt.imshow(I, cmap='gray')
    plt.show()

    I2 = pyqpi_reconstruct(I,wavelength,dx,z)
    cv2.imwrite('recovered.jpg', I2)
    plt.imshow(I2, cmap='gray')
    plt.title("recovered")
    plt.show()