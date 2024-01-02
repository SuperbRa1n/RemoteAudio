import numpy as np


def num2str(num: np.array) -> str:
    """
    将数组转换为字符串

    Args:
        num: The number to be converted

    Returns:
        A string
    """
    return "".join([str(int(i)) for i in num])


def bin2dec(num: str) -> int:
    """
    将二进制转换为十进制

    Args:
        num: The number to be converted

    Returns:
        A int
    """
    return int(num, 2)


def dec2bin(num: int, n: int) -> str:
    """
    将十进制转换为二进制

    Args:
        num: The number to be converted
        n: The length of binary

    Returns:
        A numpy array
    """
    return np.binary_repr(num, n)


def str2double(num: str) -> float:
    """
    将字符串转换为浮点数

    Args:
        num: The number to be converted

    Returns:
        A float
    """
    return float(num)


def PCMCoding(data: np.array) -> np.array:
    """
    PCM编码

    Args:
        data: The data to be encoded

    Returns:
        A numpy array
    """
    z = np.sign(data)
    maxData = np.max(np.abs(data))
    data = np.abs(data) / maxData
    q = 2048 * data
    code = np.zeros((len(data), 8))
    # 段落码判断
    for i in range(len(data)):
        if 128 <= q[i] <= 2048:
            # 在第五段与第八段之间，段位码第一位都为1
            code[i][1] = 1

        if 32 < q[i] < 128 or 512 <= q[i] <= 2048:
            # 在第三四七八段内，段位码第二位都为1
            code[i][2] = 1

        if (
            16 <= q[i] < 32
            or 64 <= q[i] < 128
            or 256 <= q[i] < 512
            or 1024 <= q[i] <= 2048
        ):
            # 在第二四六八段内，段位码第三位都为1
            code[i][3] = 1

    # 段内码判断
    N = np.zeros((len(data), 1))
    for i in range(len(data)):
        N[i] = bin2dec(num2str(code[i][1:4]))

    # 量化间隔
    a = [0, 16, 32, 64, 128, 256, 512, 1024]
    b = [1, 1, 2, 4, 8, 16, 32, 64]
    for i in range(len(data)):
        qq = int((q[i] - a[int(N[i])]) / b[int(N[i])]) + 1
        if qq == 0:
            code[i][4] = 0
            code[i][5] = 0
            code[i][6] = 0
            code[i][7] = 0
        else:
            k = dec2bin(qq, 4)
            code[i][4] = str2double(k[0])
            code[i][5] = str2double(k[1])
            code[i][6] = str2double(k[2])
            code[i][7] = str2double(k[3])

    # 符号位判断
    for i in range(len(data)):
        if z[i] == -1:
            code[i][0] = 1
        else:
            code[i][0] = 0

    code = np.reshape(code, len(data) * 8)
    return code


def PCMDecoding(data: np.array, max: float) -> np.array:
    """
    PCM解码

    Args:
        data: The data to be decoded
        max: The max value of data

    Returns:
        A numpy array
    """
    code = np.reshape(data, (int(len(data) / 8), 8))
    l = np.size(code, 0)
    a = [0, 16, 32, 64, 128, 256, 512, 1024]
    b = [1, 1, 2, 4, 8, 16, 32, 64]
    c = [
        0,
        1.5,
        2.5,
        3.5,
        4.5,
        5.5,
        6.5,
        7.5,
        8.5,
        9.5,
        10.5,
        11.5,
        12.5,
        13.5,
        14.5,
        15.5,
    ]
    k = []
    for i in range(l):
        x = code[i][0]
        T = bin2dec(num2str(code[i][1:4]))
        Y = bin2dec(num2str(code[i][4:8]))
        sign = 1 if x == 0 else -1
        if Y == 0:
            k.append(sign * a[T] / 2048)
        else:
            k.append(sign * (a[T] + b[T] * c[Y]) / 2048)

    return k * max
