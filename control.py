import numpy as np

def construction_classify(data: np.array) -> str:
    """
    构造分类器
    """
    res = ['开灯', '关灯', '开空调', '关空调']
    # 计算最小data的索引
    index = np.argmin(data)
    return res[index]