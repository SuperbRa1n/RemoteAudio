import numpy as np
from modulate import Modulate


class AWGN:
    """
    AWGN信道
    """

    def __init__(self, modulate: Modulate, SNR: float, received_signal: np.ndarray):
        self.modulate = modulate
        self.SNR = SNR
        self.received_signal = received_signal

    def P_signal(self) -> float:
        """
        计算信号功率
        """
        return np.sum(np.square(self.received_signal)) / self.modulate.N

    def noise(self) -> np.ndarray:
        """
        生成高斯白噪声
        """
        P_noise = self.P_signal() / (10 ** (self.SNR / 10))
        return np.random.normal(0, np.sqrt(P_noise), self.modulate.N)

    def output(self) -> np.ndarray:
        """
        生成AWGN信道输出
        """
        return self.received_signal + self.noise()

