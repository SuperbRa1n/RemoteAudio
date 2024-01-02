from typing import List
import numpy as np
from modulate import Modulate
from awgn import AWGN


class Demodulate:
    """
    解调器
    """

    def __init__(self, received_signal: np.ndarray, modulate: Modulate, awgn: AWGN):
        self.received_signal = received_signal
        self.N = len(received_signal)
        self.modulate = modulate
        self.awgn = awgn
        self.Eb = self.modulate.Rb * self.modulate.M / self.modulate.Rs  # 每比特能量Eb = Rb * M / Rs
        self.E = self.Eb * self.modulate.Rs  # 信号能量E = Eb * Rs

    def base_vector(self) -> np.ndarray:
        """
        生成基向量
        """
        # 生成持续时间为T的矩形脉冲, 其余N-sps个采样点为0
        g_T = np.ones(self.N)
        g_T[0:self.modulate.sps] = 1
        # 生成基向量
        varphi = 1 / np.sqrt(self.E) * g_T * np.cos(2 * np.pi * self.modulate.fc * self.modulate.t)
        return varphi

    def multiplier(self) -> np.ndarray:
        """
        乘法器
        """
        return self.received_signal * self.base_vector() + self.awgn.noise()

    def integrator(self) -> np.ndarray:
        """
        积分器
        """
        return np.convolve(self.multiplier(), np.ones(self.modulate.sps)) / self.modulate.sps

    def sampler(self) -> np.ndarray:
        """
        采样器
        """
        return self.integrator()[self.modulate.sps - 1::self.modulate.sps]

    def decision(self) -> List[int]:
        """
        判决器
        """
        x = np.sign(self.sampler())
        return [int((1 - item) / 2) for item in x]  # 二进制转换

    def output(self) -> List[int]:
        """
        解调器输出
        """
        return self.decision()

    def error_rate(self) -> float:
        """
        计算误码率
        """
        y = self.output()
        return np.sum(np.abs([y[i] - self.modulate.x[i] for i in range(self.modulate.Symbols)])) / self.modulate.Symbols