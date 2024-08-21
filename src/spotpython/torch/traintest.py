import numpy as np
from sklearn.model_selection import KFold
import torch
from spotpython.utils.device import getDevice
from torch.utils.data import random_split
from spotpython.utils.classes import get_additional_attributes
from spotpython.hyperparameters.optimizer import optimizer_handler


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


def validate_one_epoch(net, valloader, loss_function, metric, device, task):
    val_loss = 0.0
    val_steps = 0
    total = 0
    correct = 0
    metric.reset()
    for i, data in enumerate(valloader, 0):
        # get batches
        with torch.no_grad():
            input, target = data
            input, target = input.to(device), target.to(device)
            output = net(input)
            # print(f"target: {target}")
            # print(f"output: {output}")
            if task == "regression":
                target = target.unsqueeze(1)
                if target.shape == output.shape:
                    loss = loss_function(output, target)
                else:
                    raise ValueError(f"Shapes of target and output do not match: {target.shape} vs {output.shape}")
                metric_value = metric.update(output, target)
            elif task == "classification":
                loss = loss_function(output, target)
                metric_value = metric.update(output, target)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            else:
                raise ValueError(f"Unknown task: {task}")
            val_loss += loss.cpu().numpy()
            val_steps += 1
    loss = val_loss / val_steps
    metric_value = metric.compute()
    metric_name = type(metric).__name__
    print(f"{metric_name}: {metric_value:.16f}", end=" | ")
    print(f"Loss: {loss:.16f}", end=" | ")
    if task == "classification":
        accuracy = correct / total
        print(f"Acc: {accuracy:.16f}.")
    return metric_value, loss


def evaluate_cv(
    net,
    dataset,
    shuffle=False,
    loss_function=None,
    num_workers=0,
    device=None,
    show_batch_interval=10_000,
    metric=None,
    path=None,
    task=None,
    writer=None,
    writerId=None,
):
    lr_mult_instance = net.lr_mult
    epochs_instance = net.epochs
    batch_size_instance = net.batch_size
    k_folds_instance = net.k_folds
    optimizer_instance = net.optimizer
    patience_instance = net.patience
    sgd_momentum_instance = net.sgd_momentum
    removed_attributes, net = get_removed_attributes_and_base_net(net)
    metric_values = {}
    loss_values = {}
    try:
        device = getDevice(device=device)
        net.to(device)
        optimizer = optimizer_handler(
            optimizer_name=optimizer_instance,
            params=net.parameters(),
            lr_mult=lr_mult_instance,
            sgd_momentum=sgd_momentum_instance,
        )
        kfold = KFold(n_splits=k_folds_instance, shuffle=shuffle)
        for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
            print(f"Fold: {fold + 1}")
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
            trainloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size_instance, sampler=train_subsampler, num_workers=num_workers
            )
            valloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size_instance, sampler=val_subsampler, num_workers=num_workers
            )
            # each fold starts with new weights:
            reset_weights(net)
            # Early stopping parameters
            best_val_loss = float("inf")
            counter = 0
            for epoch in range(epochs_instance):
                print(f"Epoch: {epoch+1}", end=" | ")
                # training loss from one epoch:
                training_loss = train_one_epoch(
                    net=net,
                    trainloader=trainloader,
                    batch_size=batch_size_instance,
                    loss_function=loss_function,
                    optimizer=optimizer,
                    device=device,
                    show_batch_interval=show_batch_interval,
                    task=task,
                )
                # TODO: scheduler.step()
                # Early stopping check. Calculate validation loss from one epoch:
                metric_values[fold], loss_values[fold] = validate_one_epoch(
                    net, valloader=valloader, loss_function=loss_function, metric=metric, device=device, task=task
                )
                # Log the running loss averaged per batch
                metric_name = "Metric"
                if metric is not None:
                    metric_name = type(metric).__name__
                    # print(f"{metric_name} value on hold-out data: {metric_values[fold]}")
                if writer is not None:
                    writer.add_scalars(
                        "evaluate_cv fold:" + str(fold + 1) + ". Train & Val Loss and Val Metric" + writerId,
                        {"Train loss": training_loss, "Val loss": loss_values[fold], metric_name: metric_values[fold]},
                        epoch + 1,
                    )
                    writer.flush()
                if loss_values[fold] < best_val_loss:
                    best_val_loss = loss_values[fold]
                    counter = 0
                    # save model:
                    if path is not None:
                        torch.save(net.state_dict(), path)
                else:
                    counter += 1
                    if counter >= patience_instance:
                        print(f"Early stopping at epoch {epoch}")
                        break
        df_eval = sum(loss_values.values()) / len(loss_values.values())
        df_metrics = sum(metric_values.values()) / len(metric_values.values())
        df_preds = np.nan
    except Exception as err:
        print(f"Error in Net_Core. Call to evaluate_cv() failed. {err=}, {type(err)=}")
        df_eval = np.nan
        df_metrics = np.nan
        df_preds = np.nan
    add_attributes(net, removed_attributes)
    if writer is not None:
        writer.flush()
    return df_eval, df_preds, df_metrics


def evaluate_hold_out(
    net,
    train_dataset,
    shuffle,
    test_dataset=None,
    loss_function=None,
    metric=None,
    device=None,
    show_batch_interval=10_000,
    path=None,
    task=None,
    writer=None,
    writerId=None,
):
    lr_mult_instance = net.lr_mult
    epochs_instance = net.epochs
    batch_size_instance = net.batch_size
    optimizer_instance = net.optimizer
    patience_instance = net.patience
    sgd_momentum_instance = net.sgd_momentum
    removed_attributes, net = get_removed_attributes_and_base_net(net)
    try:
        device = getDevice(device=device)
        net.to(device)
        optimizer = optimizer_handler(
            optimizer_name=optimizer_instance,
            params=net.parameters(),
            lr_mult=lr_mult_instance,
            sgd_momentum=sgd_momentum_instance,
        )
        if test_dataset is None:
            trainloader, valloader = create_train_val_data_loaders(
                dataset=train_dataset, batch_size=batch_size_instance, shuffle=shuffle
            )
        else:
            trainloader, valloader = create_train_test_data_loaders(
                dataset=train_dataset, batch_size=batch_size_instance, shuffle=shuffle, test_dataset=test_dataset
            )
        # TODO: scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        # Early stopping parameters
        best_val_loss = float("inf")
        counter = 0
        # We only have "one fold" which is trained for several epochs
        # (we do not have to reset the weights for each fold):
        for epoch in range(epochs_instance):
            print(f"Epoch: {epoch + 1}", end=" | ")
            # training loss from one epoch:
            training_loss = train_one_epoch(
                net=net,
                trainloader=trainloader,
                batch_size=batch_size_instance,
                loss_function=loss_function,
                optimizer=optimizer,
                device=device,
                show_batch_interval=show_batch_interval,
                task=task,
            )
            # TODO: scheduler.step()
            # Early stopping check. Calculate validation loss from one epoch:
            metric_val, val_loss = validate_one_epoch(
                net, valloader=valloader, loss_function=loss_function, metric=metric, device=device, task=task
            )
            # Log the running loss averaged per batch
            metric_name = "Metric"
            if metric is not None:
                metric_name = type(metric).__name__
            if writer is not None:
                writer.add_scalars(
                    "evaluate_hold_out: Train & Val Loss and Val Metric" + writerId,
                    {"Train loss": training_loss, "Val loss": val_loss, metric_name: metric_val},
                    epoch + 1,
                )
                writer.flush()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                # save model:
                if path is not None:
                    torch.save(net.state_dict(), path)
            else:
                counter += 1
                if counter >= patience_instance:
                    print(f"Early stopping at epoch {epoch}")
                    break
        df_eval = val_loss
        df_preds = np.nan
    except Exception as err:
        print(f"Error in Net_Core. Call to evaluate_hold_out() failed. {err=}, {type(err)=}")
        df_eval = np.nan
        df_preds = np.nan
    add_attributes(net, removed_attributes)
    if writer is not None:
        writer.flush()
    print(f"Returned to Spot: Validation loss: {df_eval}")
    return df_eval, df_preds


def create_train_val_data_loaders(dataset, batch_size, shuffle, num_workers=0):
    test_abs = int(len(dataset) * 0.6)
    train_subset, val_subset = random_split(dataset, [test_abs, len(dataset) - test_abs])
    trainloader = torch.utils.data.DataLoader(
        train_subset, batch_size=int(batch_size), shuffle=shuffle, num_workers=num_workers
    )
    valloader = torch.utils.data.DataLoader(
        val_subset, batch_size=int(batch_size), shuffle=shuffle, num_workers=num_workers
    )
    return trainloader, valloader


def create_train_test_data_loaders(dataset, batch_size, shuffle, test_dataset, num_workers=0):
    trainloader = torch.utils.data.DataLoader(
        dataset, batch_size=int(batch_size), shuffle=shuffle, num_workers=num_workers
    )
    testloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=int(batch_size), shuffle=shuffle, num_workers=num_workers
    )
    return trainloader, testloader


def train_one_epoch(
    net,
    trainloader,
    batch_size,
    loss_function,
    optimizer,
    device,
    show_batch_interval=10_000,
    task=None,
):
    running_loss = 0.0
    epoch_steps = 0
    for batch_nr, data in enumerate(trainloader, 0):
        input, target = data
        input, target = input.to(device), target.to(device)
        optimizer.zero_grad()
        output = net(input)
        if task == "regression":
            target = target.unsqueeze(1)
            if target.shape == output.shape:
                loss = loss_function(output, target)
            else:
                raise ValueError(f"Shapes of target and output do not match: {target.shape} vs {output.shape}")
        elif task == "classification":
            loss = loss_function(output, target)
        else:
            raise ValueError(f"Unknown task: {task}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()
        epoch_steps += 1
        if batch_nr % show_batch_interval == (show_batch_interval - 1):  # print every show_batch_interval mini-batches
            print(
                "Batch: %5d. Batch Size: %d. Training Loss (running): %.3f"
                % (batch_nr + 1, int(batch_size), running_loss / epoch_steps)
            )
            running_loss = 0.0
    return loss.item()


def train_tuned(
    net,
    train_dataset,
    shuffle,
    loss_function,
    metric,
    device=None,
    show_batch_interval=10_000,
    path=None,
    task=None,
    writer=None,
):
    evaluate_hold_out(
        net=net,
        train_dataset=train_dataset,
        shuffle=shuffle,
        test_dataset=None,
        loss_function=loss_function,
        metric=metric,
        device=device,
        show_batch_interval=show_batch_interval,
        path=path,
        task=task,
        writer=writer,
    )


def test_tuned(net, shuffle, test_dataset=None, loss_function=None, metric=None, device=None, path=None, task=None):
    batch_size_instance = net.batch_size
    removed_attributes, net = get_removed_attributes_and_base_net(net)
    if path is not None:
        net.load_state_dict(torch.load(path))
        net.eval()
    try:
        device = getDevice(device=device)
        net.to(device)
        valloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=int(batch_size_instance), shuffle=shuffle, num_workers=0
        )
        metric_value, loss = validate_one_epoch(
            net, valloader=valloader, loss_function=loss_function, metric=metric, device=device, task=task
        )
        df_eval = loss
        df_metric = metric_value
        df_preds = np.nan
    except Exception as err:
        print(f"Error in Net_Core. Call to test_tuned() failed. {err=}, {type(err)=}")
        df_eval = np.nan
        df_metric = np.nan
        df_preds = np.nan
    add_attributes(net, removed_attributes)
    print(f"Final evaluation: Validation loss: {df_eval}")
    print(f"Final evaluation: Validation metric: {df_metric}")
    print("----------------------------------------------")
    return df_eval, df_preds, df_metric
