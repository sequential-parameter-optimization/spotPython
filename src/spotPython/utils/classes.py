import torch.nn as nn
import inspect


def get_additional_attributes(b):
    a = nn.Module()
    additional_attributes = set(vars(b)) - set(vars(a))
    print(f"Additional attributes: {additional_attributes}")
    return additional_attributes


def get_additional_methods(b):
    a = nn.Module()
    additional_methods = set(
        [method_name for method_name, method in inspect.getmembers(b, predicate=inspect.ismethod)]
    ) - set([method_name for method_name, method in inspect.getmembers(a, predicate=inspect.ismethod)])
    print(f"Additional methods: {additional_methods}")
    return additional_methods
