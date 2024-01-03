import numpy as np
import matplotlib.pyplot as plt
import AudioReg as ar
import pcm as sc
from modulate import Modulate
from demodulate import Demodulate
from awgn import AWGN
import mfcc
import control as ct

path = './assets/'

Rb = 50  # 比特速率50 bps
M = 2  # 二进制BPSK
Rs = Rb * np.log2(M)  # 码元速率50 bps * log2(2) = 50 bps * 1 = 50 bps
fc = 5e3 # 载波频率5 kHz
fs = 10e3  # 采样频率100 kHz
sps = fs / Rs  # 每个码元采样点数100 kHz / 50 bps = 2000
SNR = 10  # 信噪比10 dB

def main():
    # 读取音频数据
    audio1 = ar.read(path + '开灯1.WAV')
    audio2 = ar.read(path + '开灯2.WAV')
    audio3 = ar.read(path + '开空调1.WAV')
    audio4 = ar.read(path + '开空调2.WAV')
    audio5 = ar.read(path + '关灯1.WAV')
    audio6 = ar.read(path + '关灯2.WAV')
    audio7 = ar.read(path + '关空调1.WAV')
    audio8 = ar.read(path + '关空调2.WAV')
    mf = mfcc.MFCC(fs, 512, 26, 13, 0.97, 0, 4000)
    dtw_mat = np.zeros((4, 4))
    train_set = [audio1, audio3, audio5, audio7]
    test_set = [audio2, audio4, audio6, audio8]
    res_set = []
    for i in range(4):
        plt.figure(i)
        plt.subplot(711)
        plt.plot(test_set[i])
        plt.xlabel('Time(s)')
        plt.ylabel('Amplitude')
        plt.title('Original Audio')
        # 信源编码
        audio_encoded = sc.PCMCoding(test_set[i])
        # 实例化调制器
        mod = Modulate(Rb, M, fc, fs, audio_encoded)
        # BPSK调制
        y = mod.BPSK()
        # 绘制调制后的波形
        plt.subplot(712)
        plt.plot(mod.t, y)
        plt.xlabel('Time(s)')
        plt.ylabel('Amplitude')
        plt.title('BPSK Modulation')
        # 绘制信号的时域波形
        plt.subplot(713)
        plt.plot(mod.t, mod.baseband_shaping_filter())
        plt.xlabel('Time(s)')
        plt.ylabel('Amplitude')
        plt.title('Baseband Shaping Filter')
        # 通过AWGN信道
        awgn = AWGN(mod, SNR, y)
        y = awgn.output()
        # 绘制通过AWGN信道后的波形
        plt.subplot(721)
        plt.plot(mod.t, y)
        plt.xlabel('Time(s)')
        plt.ylabel('Amplitude')
        plt.title('BPSK Modulation Through AWGN Channel')
        # 解调
        demod = Demodulate(y, mod, awgn)
        y = demod.integrator()
        # 绘制采样后的波形
        plt.subplot(722)
        plt.plot(demod.sampler())
        plt.xlabel('Time(s)')
        plt.ylabel('Amplitude')
        plt.title('BPSK Sampling')
        # 绘制解调后的波形
        plt.subplot(723)
        plt.plot(demod.output())
        plt.xlabel('Time(s)')
        plt.ylabel('Amplitude')
        plt.title('BPSK Demodulation')
        # 输出解调结果
        deres = demod.output()
        # 信源解码
        audio_decoded = sc.PCMDecoding(deres, int(np.max(np.abs(audio_encoded))))
        # 绘制解码后的波形
        plt.subplot(731)
        plt.plot(audio_decoded)
        plt.xlabel('Time(s)')
        plt.ylabel('Amplitude')
        plt.title('PCM Decoding')
        # 计算误码率
        error_rate = np.sum(audio1 != audio_encoded) / len(audio1)
        print(f'误码率:{error_rate}')
        res_set.append(audio_decoded)
    for i in range(4):
        for j in range(4):
            dtw_mat[i, j] = mfcc.dtw(mf, train_set[i], res_set[j])
    dtw_mat = mfcc.data_pro(dtw_mat)
    print(dtw_mat)
    plt.show()
    # 指令控制测试
    for i in range(4):
        print(ct.construction_classify(dtw_mat[i]))
    
if __name__ == '__main__':
    main()