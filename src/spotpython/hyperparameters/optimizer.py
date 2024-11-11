import torch.optim
from typing import Any, Union


def optimizer_handler(optimizer_name: str, params: Union[list, torch.Tensor], lr_mult: float = 1.0, **kwargs: Any) -> torch.optim.Optimizer:
    """Returns an instance of the specified optimizer. See Notes below for supported optimizers.

    Args:
        optimizer_name (str):
            The name of the optimizer to use.
        params (list or torch.Tensor):
            The parameters to optimize.
        lr_mult (float, optional):
            A multiplier for the learning rate. Defaults to 1.0.
        **kwargs:
            Additional keyword arguments for the optimizer.

    Notes:
        The following optimizers are supported (see also: https://pytorch.org/docs/stable/optim.html#base-class):

            * Adadelta
            * Adagrad
            * Adam
            * AdamW
            * SparseAdam
            * ASGD
            * LBFGS
            * NAdam
            * RAdam
            * RMSprop
            * Rprop
            * SGD

    Returns:
        (torch.optim.Optimizer):
            An instance of the specified optimizer.

    Examples:
        >>> from torch.utils.data import DataLoader
            from spotpython.data.diabetes import Diabetes
            from spotpython.light.netlightregression import NetLightRegression
            from torch import nn
            import lightning as L
            BATCH_SIZE = 8
            lr_mult=0.1
            dataset = Diabetes()
            train_loader = DataLoader(dataset, batch_size=BATCH_SIZE)
            test_loader = DataLoader(dataset, batch_size=BATCH_SIZE)
            val_loader = DataLoader(dataset, batch_size=BATCH_SIZE)
            # First example: Adam
            net_light_base = NetLightRegression(l1=128, epochs=10, batch_size=BATCH_SIZE,
                                            initialization='xavier', act_fn=nn.ReLU(),
                                            optimizer='Adam', dropout_prob=0.1, lr_mult=lr_mult,
                                            patience=5, _L_in=10, _L_out=1)
            trainer = L.Trainer(max_epochs=2,  enable_progress_bar=False)
            trainer.fit(net_light_base, train_loader)
            # Adam uses a lr which is calculated as lr=lr_mult * 0.001, so this value
            # should be 0.1 * 0.001 = 0.0001
            trainer.optimizers[0].param_groups[0]["lr"] == lr_mult*0.001
            # Second example: Adadelta
            net_light_base = NetLightRegression(l1=128, epochs=10, batch_size=BATCH_SIZE,
                                            initialization='xavier', act_fn=nn.ReLU(),
                                            optimizer='Adadelta', dropout_prob=0.1, lr_mult=lr_mult,
                                            patience=5, _L_in=10, _L_out=1)
            trainer = L.Trainer(max_epochs=2,  enable_progress_bar=False)
            trainer.fit(net_light_base, train_loader)
            # Adadelta uses a lr which is calculated as lr=lr_mult * 1.0, so this value
            # should be 1.0 * 0.1 = 0.1
            trainer.optimizers[0].param_groups[0]["lr"] == lr_mult*1.0
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
