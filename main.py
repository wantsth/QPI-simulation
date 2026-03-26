import numpy as np
import matplotlib.pyplot as plt
import cv2
from pyDHM import numericalPropagation
from keras.datasets import mnist

# def load_mnist():
#     (x_train, _), _ = mnist.load_data()
#     img = x_train[0]   # 第一张
#     img = img / 255.0
#     plt. imshow(img,cmap='gray')
#     plt.show()
#     return img

def image_process(name):
    image = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
    matrix = image / 255.0

    return matrix

def Transmission(U_in, matrix, wavelength, delta_n):
    h_max = 1e-6   # 1μm,最大厚度
    h = matrix * h_max  

    phi = (2*np.pi / wavelength) * delta_n * h
    T = np.exp(1j * phi)

    return U_in * T

if __name__ == "__main__":

    wavelength = 632.8e-9   # 波长
    z = 5e-3    # 传播距离
    dx = 2e-6   # 折射率差
    delta_n = 0.01  # 像素间距

    matrix = image_process('cameraman.jpg')
    # matrix = load_mnist()
    # scale = 8
    # matrix = np.kron(matrix, np.ones((scale, scale)))  # 放大图像，增加采样
    h, w = matrix.shape

    x = (np.arange(w) - w//2) * dx
    y = (np.arange(h) - h//2) * dx
    X, Y = np.meshgrid(x, y)        # 置于中心

    U_in = np.ones((h, w), dtype=complex)   #入射光/参考光

    U_sample = Transmission(U_in, matrix, wavelength, delta_n)  #样本光，刚离开样本

    U_prop = numericalPropagation.angularSpectrum(U_sample, z, wavelength, dx, dx)  #样本光传播，至与参考光干涉

    fx0 = 1 / (8 * dx)  # 空间频率，sin（θ）/λ
    U_ref = np.exp(1j * 2*np.pi * fx0 * X)  

    I = np.abs(U_prop + U_ref)**2
    # 神秘调参？
    # ===== 加一点噪声（更真实）=====
   # I = I + 0.01 * np.random.randn(h, w)
    # ===== 归一化到相机 =====
    I = I - I.min()
    I = I / I.max()

    plt.imshow(I, cmap='gray')
    plt.title("Off-axis hologram")
    plt.axis('off')
    plt.show()

    cv2.imwrite('hologram.jpg', (I*255).astype(np.uint8))

    # ===== 频谱检查（强烈建议看！）=====
    F = np.fft.fftshift(np.fft.fft2(I))
    plt.imshow(np.log(np.abs(F)+1), cmap='gray')
    plt.title("Fourier Spectrum")
    plt.axis('off')
    plt.show()