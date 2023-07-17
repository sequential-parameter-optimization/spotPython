import torch.optim
from typing import Any, Union


def optimizer_handler(
    optimizer_name: str, params: Union[list, torch.Tensor], lr_mult: float = 1.0, **kwargs: Any
) -> torch.optim.Optimizer:
    """Returns an instance of the specified optimizer.

    Args:
        optimizer_name (str): The name of the optimizer to use.
        params (list or torch.Tensor): The parameters to optimize.
        lr_mult (float, optional): A multiplier for the learning rate. Defaults to 1.0.
        **kwargs: Additional keyword arguments for the optimizer.

    Returns:
        (torch.optim.Optimizer):
            An instance of the specified optimizer.

    Examples:
        >>> model = torch.nn.Linear(10, 1)
        >>> optimizer = optimizer_handler("Adadelta", model.parameters(), lr_mult=0.5)
        >>> print(optimizer)
        Adadelta (
            Parameter Group 0
                eps: 1e-06
                lr: 0.5
                rho: 0.9
                weight_decay: 0
        )
    """
    if optimizer_name == "Adadelta":
        return torch.optim.Adadelta(
            params,
            lr=lr_mult * 1.0,
            rho=0.9,
            eps=1e-06,
            weight_decay=0,
            foreach=None,
            maximize=False,
            # differentiable=False,
        )
    elif optimizer_name == "Adagrad":
        return torch.optim.Adagrad(
            params,
            lr=lr_mult * 0.01,
            lr_decay=0,
            weight_decay=0,
            initial_accumulator_value=0,
            eps=1e-10,
            foreach=None,
            maximize=False,
            # differentiable=False,
        )
    elif optimizer_name == "Adam":
        return torch.optim.Adam(
            params,
            lr=lr_mult * 0.001,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0,
            amsgrad=False,
            foreach=None,
            maximize=False,
            capturable=False,
            # differentiable=False,
            fused=None,
        )
    elif optimizer_name == "AdamW":
        return torch.optim.AdamW(
            params,
            lr=lr_mult * 0.001,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0.01,
            amsgrad=False,
            foreach=None,
            maximize=False,
            capturable=False,
            # differentiable=False,
            # fused=None,
        )
    elif optimizer_name == "SparseAdam":
        return torch.optim.SparseAdam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, maximize=False)
    elif optimizer_name == "Adamax":
        return torch.optim.Adamax(
            params,
            lr=lr_mult * 0.002,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0,
            foreach=None,
            maximize=False,
            # differentiable=False,
        )
    elif optimizer_name == "ASGD":
        return torch.optim.ASGD(
            params,
            lr=lr_mult * 0.01,
            lambd=0.0001,
            alpha=0.75,
            t0=1000000.0,
            weight_decay=0,
            foreach=None,
            maximize=False,
            # differentiable=False,
        )
    elif optimizer_name == "LBFGS":
        return torch.optim.LBFGS(
            params,
            lr=lr_mult * 1,
            max_iter=20,
            max_eval=None,
            tolerance_grad=1e-07,
            tolerance_change=1e-09,
            history_size=100,
            line_search_fn=None,
        )
    elif optimizer_name == "NAdam":
        return torch.optim.NAdam(
            params,
            lr=lr_mult * 0.002,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0,
            momentum_decay=0.004,
            foreach=None,
            # differentiable=False,
        )
    elif optimizer_name == "RAdam":
        return torch.optim.RAdam(
            params,
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0,
            foreach=None,
            # differentiable=False
        )
    elif optimizer_name == "RMSprop":
        return torch.optim.RMSprop(
            params,
            lr=lr_mult * 0.01,
            alpha=0.99,
            eps=1e-08,
            weight_decay=0,
            momentum=0,
            centered=False,
            foreach=None,
            maximize=False,
            # differentiable=False,
        )
    elif optimizer_name == "Rprop":
        return torch.optim.Rprop(
            params,
            lr=lr_mult * 0.01,
            etas=(0.5, 1.2),
            step_sizes=(1e-06, 50),
            foreach=None,
            maximize=False,
            # differentiable=False,
        )
    elif optimizer_name == "SGD":
        return torch.optim.SGD(
            params,
            lr=lr_mult * 1e-3,
            momentum=0,
            dampening=0,
            weight_decay=0,
            nesterov=False,
            maximize=False,
            foreach=None,
            # differentiable=False,
        )
    else:
        raise ValueError(f"Optimizer {optimizer_name} not supported")
