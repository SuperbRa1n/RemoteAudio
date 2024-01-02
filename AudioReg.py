import os
import numpy as np
from pydub import AudioSegment
import matplotlib.pyplot as plt

def read(wav_file: str) -> np.array:
    """
    读取wav文件

    Args:
        wav_file: The path of wav file

    Returns:
        A numpy array
    """
    if not os.path.exists(wav_file):
        raise ValueError("wav file {} not exists".format(wav_file))
    audio_data = AudioSegment.from_wav(wav_file)
    audio_data = audio_data.set_channels(1)
    audio_data = audio_data.set_frame_rate(16000)
    return np.array(audio_data.get_array_of_samples())


if __name__ == '__main__':
    audio = read('demo.wav')
    # 绘制时域波形
    t = np.arange(0, len(audio)) * (1.0 / 16000)
    plt.plot(t, audio)
    plt.show()




