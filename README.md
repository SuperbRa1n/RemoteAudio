# RemoteAudio
一个远程智能家居声控系统
## 运行
首先在python>=3.8环境下安装`requirements.txt`中需要的modules：
```cmd
pip install -r requirements.txt
```
再运行`main.py`文件：
```cmd
python main.py
```

## 文件架构说明
* `assets`：存放测试用的音频`.WAV`格式文件
* `results`：存放绘制的图片结果
* `AudioReg.py`：语音信号采集
* `pcm.py`：信源PCM-A律十三折线编码
* `modulate.py`：调制器，采用BPSK调制
* `awgn.py`：模拟加性高斯白噪声信道，信噪比$\mathrm{SNR}$可控
* `demodulate.py`：解调器，采用相干解调
* `mfcc.py`：梅尔频率倒谱系数的求解，用于信号的识别与分类
* `control.py`：控制模块，用于将数据转化为可执行的指令
* `main.py`：主函数，按顺序执行所有内容，实现整个系统

## 报告指路
https://tannin-1316822731.cos.ap-nanjing.myqcloud.com/2024-01-03-%E4%BF%A1%E6%8E%A7%E8%AE%A1%E5%A4%A7%E4%BD%9C%E4%B8%9A.pdf