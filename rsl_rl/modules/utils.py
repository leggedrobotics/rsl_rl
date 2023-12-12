from torch import nn


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    elif act_name == "linear":
        return nn.Identity()
    elif act_name == "softmax":
        return nn.Softmax()
    else:
        print("invalid activation function!")
        return None


def init_xavier_uniform(layer, activation):
    try:
        nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain(activation))
    except ValueError:
        nn.init.xavier_uniform_(layer.weight)
