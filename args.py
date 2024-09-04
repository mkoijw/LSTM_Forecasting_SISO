from collections import namedtuple

def args_parser():
    Args = namedtuple('Args', ['input_size', 'hidden_size', 'num_layers', 'output_size', 'batch_size', 'optimizer','lr','weight_decay','step_size','gamma','epochs'])
    args = Args(
        input_size = 1,
        hidden_size = 10,
        num_layers = 1,
        output_size = 1,
        batch_size = 10,
        optimizer = 'adam',
        lr = 0.001,
        weight_decay = 1e-5,
        step_size = 10,
        gamma = 0.7,
        epochs = 50
    )
    return args
# class Args:
#     input_size = 1,
#     hidden_size = 10,
#     num_layers = 1,
#     output_size = 1,
#     batch_size = 554,
#     optimizer = 'adam',
#     lr = 0.001,
#     weight_decay = 1e-5
#     step_size = 10
#     gamma = 0.7
#
# args = Args()