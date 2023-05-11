import torch.nn as nn


def get_additional_attributes(b):
    a = nn.Module()
    additional_attributes = set(vars(b)) - set(vars(a))
    print(f"Additional attributes: {additional_attributes}")
    return additional_attributes
