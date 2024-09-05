from collections import namedtuple

def args_parser():
    Args = namedtuple('Args', ['input_size', 'hidden_size', 'num_layers', 'output_size', 'batch_size', 'optimizer','lr','weight_decay','step_size','gamma','epochs'])
    args = Args(
        input_size = 1,
        hidden_size = 5,
        num_layers = 1,
        output_size = 1,
        batch_size = 20,
        optimizer = 'adam',
        lr = 0.01,#学习率，控制每次参数更新的步长大小
        weight_decay = 1e-5,#L2正则化的系数（也叫权重衰减）。它用于在优化过程中对权重施加惩罚，防止过拟合
        step_size = 5,#每隔多少个训练周期（epochs）就更新一次学习率
        gamma = 0.9,#缩放因子，如果 gamma 设置为 0.1，那么每隔 step_size 个周期，学习率将变成原来的 0.1 倍。
        epochs = 20
    )
    return args
