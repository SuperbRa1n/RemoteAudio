import numpy as np


class Modulate:
    """
    调制器
    """

    def __init__(self, Rb: float, M: int, fc: float, fs: float, x: list):
        self.Rb = Rb
        self.M = M
        self.Rs = Rb * np.log2(M)
        self.fc = fc
        self.fs = fs
        self.sps = int(fs / self.Rs)
        self.x = x
        self.Symbols = len(x)
        self.N = int(self.Symbols * self.sps)
        self.t = np.arange(0, self.N) / fs

    def baseband_shaping_filter(self) -> np.ndarray:
        """
        生成基带成型滤波器
        """
        x = np.repeat(self.x, self.sps)
        return x

    def BPSK(self) -> np.ndarray:
        # 基带成型滤波
        x = self.baseband_shaping_filter()
        # 生成载波
        carrier = np.cos(2 * np.pi * self.fc * self.t + np.pi * x)
        return carrier