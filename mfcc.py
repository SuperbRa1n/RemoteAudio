from turtle import distance
import numpy as np


class MFCC:
    """
    MFCC特征提取器
    """

    def __init__(
        self,
        fs: int,
        NFFT: int,
        NFB: int,
        NCEP: int,
        pre_emphasis: float,
        low_freq: int,
        high_freq: int,
    ):
        self.fs = fs
        self.NFFT = NFFT
        self.NFB = NFB
        self.NCEP = NCEP
        self.pre_emphasis = pre_emphasis
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.mel_points = self.mel_points()
        self.mel_filter = self.mel_filter()
        self.dct_filter = self.dct_filter()

    def mel_points(self) -> np.ndarray:
        """
        计算mel刻度
        """
        low_mel = self.freq2mel(self.low_freq)
        high_mel = self.freq2mel(self.high_freq)
        return np.linspace(low_mel, high_mel, self.NFB + 2)

    def freq2mel(self, freq: float) -> float:
        """
        频率到mel刻度的转换
        """
        return 2595 * np.log10(1 + freq / 700)

    def mel2freq(self, mel: float) -> float:
        """
        mel刻度到频率的转换
        """
        return 700 * (10 ** (mel / 2595) - 1)

    def mel_filter(self) -> np.ndarray:
        """
        计算mel滤波器
        """
        mel_filter = np.zeros((self.NFB, int(self.NFFT / 2 + 1)))
        for i in range(1, self.NFB + 1):
            for j in range(int(self.NFFT / 2 + 1)):
                if self.mel_points[i - 1] <= j < self.mel_points[i]:
                    mel_filter[i - 1, j] = (j - self.mel_points[i - 1]) / (
                        self.mel_points[i] - self.mel_points[i - 1]
                    )
                elif self.mel_points[i] <= j <= self.mel_points[i + 1]:
                    mel_filter[i - 1, j] = (self.mel_points[i + 1] - j) / (
                        self.mel_points[i + 1] - self.mel_points[i]
                    )
        return mel_filter

    def dct_filter(self) -> np.ndarray:
        """
        计算DCT滤波器
        """
        dct_filter = np.zeros((self.NCEP, self.NFB))
        for i in range(self.NCEP):
            for j in range(self.NFB):
                dct_filter[i, j] = np.cos(np.pi * i / self.NFB * (j + 0.5))
        return dct_filter

    def pre_emphasis_filter(self, data: np.ndarray) -> np.ndarray:
        """
        预加重滤波器
        """
        return np.append(data[0], data[1:] - self.pre_emphasis * data[:-1])

    def hamming_window(self, data: np.ndarray) -> np.ndarray:
        """
        汉明窗
        """
        return data * np.hamming(len(data))

    def power_spectrum(self, data: np.ndarray) -> np.ndarray:
        """
        功率谱
        """
        return np.square(np.abs(np.fft.rfft(data, self.NFFT)))

    def mel_spectrum(self, data: np.ndarray) -> np.ndarray:
        """
        mel谱
        """
        return np.dot(self.mel_filter, data)

    def log_spectrum(self, data: np.ndarray) -> np.ndarray:
        """
        对数谱
        """
        return np.log(data)

    def dct_spectrum(self, data: np.ndarray) -> np.ndarray:
        """
        DCT谱
        """
        return np.dot(self.dct_filter, data)

    def output(self, data: np.ndarray) -> np.ndarray:
        """
        MFCC输出
        """
        data = self.pre_emphasis_filter(data)
        data = self.hamming_window(data)
        data = self.power_spectrum(data)
        data = self.mel_spectrum(data)
        data = self.log_spectrum(1 + data)
        data = self.dct_spectrum(data)
        return data

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """
        归一化
        """
        return (data - np.mean(data)) / np.std(data)

    def mfcc(self, data: np.ndarray) -> np.ndarray:
        """
        MFCC特征提取
        """
        data = self.output(data)
        data = self.normalize(data)
        return data


def dtw(mfcc: MFCC, data1: np.ndarray, data2: np.ndarray) -> float:
    """
    DTW距离
    """
    data1 = mfcc.mfcc(data1)
    data2 = mfcc.mfcc(data2)
    distance = np.zeros((len(data1), len(data2)))
    distance[0, 0] = np.sqrt(np.sum(np.square(data1[0] - data2[0])))
    for i in range(1, len(data1)):
        distance[i, 0] = (
            np.sqrt(np.sum(np.square(data1[i] - data2[0]))) + distance[i - 1, 0]
        )
    for j in range(1, len(data2)):
        distance[0, j] = (
            np.sqrt(np.sum(np.square(data1[0] - data2[j]))) + distance[0, j - 1]
        )
    for i in range(1, len(data1)):
        for j in range(1, len(data2)):
            distance[i, j] = np.sqrt(np.sum(np.square(data1[i] - data2[j]))) + min(
                distance[i - 1, j], distance[i, j - 1], distance[i - 1, j - 1]
            )
    return distance[-1, -1]


def data_pro(data: np.ndarray) -> np.ndarray:
    """
    数据预处理
    """
    for i in range(4):
        min = np.min(data[i])
        for j in range(4):
            if data[i][j] == min:
                data[i][j], data[i][i] = data[i][i], data[i][j]
    
    return data