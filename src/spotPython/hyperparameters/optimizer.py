import torch.optim


def optimizer_handler(optimizer_name: str, params, sgd_lr=0.9, **kwargs):
    if optimizer_name == "Adadelta":
        return torch.optim.Adadelta(
            params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0, foreach=None, maximize=False, differentiable=False
        )
    elif optimizer_name == "Adagrad":
        return torch.optim.Adagrad(
            params,
            lr=0.01,
            lr_decay=0,
            weight_decay=0,
            initial_accumulator_value=0,
            eps=1e-10,
            foreach=None,
            maximize=False,
            differentiable=False,
        )
    elif optimizer_name == "Adam":
        return torch.optim.Adam(
            params,
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0,
            amsgrad=False,
            foreach=None,
            maximize=False,
            capturable=False,
            differentiable=False,
            fused=None,
        )
    elif optimizer_name == "AdamW":
        return torch.optim.AdamW(
            params,
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0.01,
            amsgrad=False,
            foreach=None,
            maximize=False,
            capturable=False,
            differentiable=False,
            fused=None,
        )
    elif optimizer_name == "SparseAdam":
        return torch.optim.SparseAdam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, maximize=False)
    elif optimizer_name == "Adamax":
        return torch.optim.Adamax(
            params,
            lr=0.002,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0,
            foreach=None,
            maximize=False,
            differentiable=False,
        )
    elif optimizer_name == "ASGD":
        return torch.optim.ASGD(
            params,
            lr=0.01,
            lambd=0.0001,
            alpha=0.75,
            t0=1000000.0,
            weight_decay=0,
            foreach=None,
            maximize=False,
            differentiable=False,
        )
    elif optimizer_name == "LBFGS":
        return torch.optim.LBFGS(
            params,
            lr=1,
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
            lr=0.002,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0,
            momentum_decay=0.004,
            foreach=None,
            differentiable=False,
        )
    elif optimizer_name == "RAdam":
        return torch.optim.RAdam(
            params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, foreach=None, differentiable=False
        )
    elif optimizer_name == "RMSprop":
        return torch.optim.RMSprop(
            params,
            lr=0.01,
            alpha=0.99,
            eps=1e-08,
            weight_decay=0,
            momentum=0,
            centered=False,
            foreach=None,
            maximize=False,
            differentiable=False,
        )
    elif optimizer_name == "Rprop":
        return torch.optim.Rprop(
            params, lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-06, 50), foreach=None, maximize=False, differentiable=False
        )
    elif optimizer_name == "SGD":
        return torch.optim.SGD(
            params,
            lr=sgd_lr,
            momentum=0,
            dampening=0,
            weight_decay=0,
            nesterov=False,
            maximize=False,
            foreach=None,
            differentiable=False,
        )
    else:
        raise ValueError(f"Optimizer {optimizer_name} not supported")
