import tqdm
import numpy as np
from scipy import fft
from skimage import io, exposure, img_as_ubyte, img_as_float

def firstOrderDerivative(n, k=1):
    return np.eye(n) * (-1) + np.eye(n, k=k)


def toeplitizMatrix(n, row):
    vecDD = np.zeros(n)
    vecDD[0] = 4
    vecDD[1] = -1
    vecDD[row] = -1
    vecDD[-1] = -1
    vecDD[-row] = -1
    return vecDD


def vectorize(matrix):
    return matrix.T.ravel()


def reshape(vector, row, col):
    return vector.reshape((row, col), order='F')


class LIME:
    def __init__(self, iterations=10, alpha=2, rho=1.5, gamma=0.7, strategy=2, *args, **kwargs):
        self.iterations = iterations
        self.alpha = alpha
        self.rho = rho
        self.gamma = gamma
        self.strategy = strategy

    def load(self, imgPath):
        if isinstance(imgPath, str):
            self.L = img_as_float(io.imread(imgPath))
        elif isinstance(imgPath, np.ndarray):
            self.L = img_as_float(imgPath)
        else:
            print(f"The input should be path to image of numpy array.")

        self.row = self.L.shape[0]
        self.col = self.L.shape[1]

        self.T_hat = np.max(self.L, axis=2)
        self.dv = firstOrderDerivative(self.row)
        self.dh = firstOrderDerivative(self.col, -1)
        self.vecDD = toeplitizMatrix(self.row * self.col, self.row)
        self.W = self.weightingStrategy()

    def weightingStrategy(self):
        if self.strategy == 2:
            dTv = self.dv @ self.T_hat
            dTh = self.T_hat @ self.dh
            Wv = 1 / (np.abs(dTv) + 1)
            Wh = 1 / (np.abs(dTh) + 1)
            return np.vstack([Wv, Wh])
        else:
            return np.ones((self.row * 2, self.col))

    def __T_subproblem(self, G, Z, u):
        X = G - Z / u
        Xv = X[:self.row, :]
        Xh = X[self.row:, :]
        temp = self.dv @ Xv + Xh @ self.dh
        numerator = fft.fft(vectorize(2 * self.T_hat + u * temp))
        denominator = fft.fft(self.vecDD * u) + 2
        T = fft.ifft(numerator / denominator)
        T = np.real(reshape(T, self.row, self.col))
        return exposure.rescale_intensity(T, (0, 1), (0.001, 1))

    def __G_subproblem(self, T, Z, u, W):
        dT = self.__derivative(T)
        epsilon = self.alpha * W / u
        X = dT + Z / u
        return np.sign(X) * np.maximum(np.abs(X) - epsilon, 0)

    def __Z_subproblem(self, T, G, Z, u):
        dT = self.__derivative(T)
        return Z + u * (dT - G)

    def __u_subproblem(self, u):
        return u * self.rho

    def __derivative(self, matrix):
        v = self.dv @ matrix
        h = matrix @ self.dh
        return np.vstack([v, h])

    def illumMap(self):
        T = np.zeros((self.row, self.col))
        G = np.zeros((self.row * 2, self.col))
        Z = np.zeros((self.row * 2, self.col))
        u = 1

        for _ in tqdm.trange(0, self.iterations):
            T = self.__T_subproblem(G, Z, u)
            G = self.__G_subproblem(T, Z, u, self.W)
            Z = self.__Z_subproblem(T, G, Z, u)
            u = self.__u_subproblem(u)

        return T ** self.gamma

    def enhance(self):
        self.T = self.illumMap()
        self.R = self.L / np.repeat(self.T[:, :, np.newaxis], 3, axis=2)
        self.R = exposure.rescale_intensity(self.R, (0, 1))
        self.R = img_as_ubyte(self.R)
        return self.R
