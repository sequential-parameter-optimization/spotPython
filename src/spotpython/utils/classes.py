import torch.nn as nn
import inspect


def get_additional_attributes(b):
    a = nn.Module()
    additional_attributes = set(vars(b)) - set(vars(a))
    return additional_attributes


def get_additional_methods(b):
    a = nn.Module()
    additional_methods = set(
        [method_name for method_name, method in inspect.getmembers(b, predicate=inspect.ismethod)]
    ) - set([method_name for method_name, method in inspect.getmembers(a, predicate=inspect.ismethod)])
    return additional_methods


def remove_attributes(net, atttributes_to_remove):
    for attr in atttributes_to_remove:
        delattr(net, attr)
    return net


def reset_weights(net):
    for layer in net.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()


def add_attributes(net, attributes):
    # directly modifies the net object (no return value)
    for key, value in attributes.items():
        setattr(net, key, value)


def get_removed_attributes_and_base_net(net):
    # 1. Determine the additional attributes:
    removed_attributes = get_additional_attributes(net)
    # 2. Save the attributes:
    attributes = {}
    for attr in removed_attributes:
        attributes[attr] = getattr(net, attr)
    # 3. Remove the attributes:
    net = remove_attributes(net, removed_attributes)
    return attributes, net
