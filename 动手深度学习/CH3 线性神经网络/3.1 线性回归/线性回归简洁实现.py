"""导入必要的库"""
import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

"""生成y=Xw+b+噪声"""
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

"""构造一个PyTorch数据迭代器"""
def load_array(data_arrays, batch_size, is_train=True):  #@save
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)
batch_size = 10
data_iter = load_array((features, labels), batch_size)
