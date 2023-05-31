import numpy as np
from sklearn.model_selection import KFold
import torch
from torch import nn as nn
from spotPython.utils.device import getDevice
from torch.utils.data import random_split
from spotPython.utils.classes import get_additional_attributes
from spotPython.hyperparameters.optimizer import optimizer_handler


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


def train_fold(net, trainloader, epochs, loss_function, optimizer, device, show_batch_interval=10_000):
    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}")
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % show_batch_interval == (show_batch_interval - 1):  # print every show_batch_interval mini-batches
                print("Batch: %5d. Training Loss (running): %.3f" % (i + 1, running_loss / epoch_steps))
                running_loss = 0.0


def validate_fold_or_hold_out(net, valloader, loss_function, metric, device):
    val_loss = 0.0
    val_steps = 0
    total = 0
    correct = 0
    metric.reset()
    for i, data in enumerate(valloader, 0):
        with torch.no_grad():
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            # print(f"outputs: {outputs}")
            # print(f"labels: {labels}")
            metric_value = metric.update(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            # print(f"predicted: {predicted}")
            total += labels.size(0)
            # print(f"total: {total}")
            correct += (predicted == labels).sum().item()
            # print(f"correct: {correct}")
            #
            # print(f"predicted: {predicted}")
            # print(f"labels: {labels}")
            # metric_value = metric.update(predicted, labels).to(device)
            # print(f"Accuracy on batch {i}: {acc}")
            #
            loss = loss_function(outputs, labels)
            val_loss += loss.cpu().numpy()
            val_steps += 1
    accuracy = correct / total
    loss = val_loss / val_steps
    print(f"Loss on hold-out set: {loss}")
    print(f"Accuracy on hold-out set: {accuracy}")
    # metric on all batches using custom accumulation
    metric_value = metric.compute()
    print(f"Metric value on hold-out data: {metric_value}")
    return metric_value, loss


def evaluate_cv(
    net,
    dataset,
    shuffle=False,
    num_workers=0,
    device=None,
    show_batch_interval=10_000,
):
    lr_mult_instance = net.lr_mult
    epochs_instance = net.epochs
    batch_size_instance = net.batch_size
    k_folds_instance = net.k_folds
    loss_function_instance = net.loss_function
    optimizer_instance = net.optimizer
    sgd_momentum_instance = net.sgd_momentum
    removed_attributes, net = get_removed_attributes_and_base_net(net)
    metric_values = {}
    loss_values = {}
    try:
        device = getDevice(device=device)
        if torch.cuda.is_available():
            device = "cuda:0"
            if torch.cuda.device_count() > 1:
                print("We will use", torch.cuda.device_count(), "GPUs!")
                net = nn.DataParallel(net)
        net.to(device)
        optimizer = optimizer_handler(
            optimizer_name=optimizer_instance,
            params=net.parameters(),
            lr_mult=lr_mult_instance,
            sgd_momentum=sgd_momentum_instance,
        )
        loss_function = loss_function_instance
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
            reset_weights(net)
            # Train fold for several epochs:
            train_fold(
                net,
                trainloader,
                epochs_instance,
                loss_function,
                optimizer,
                device,
                show_batch_interval=show_batch_interval,
            )
            # Validate fold: use only loss for tuning
            metric_values[fold], loss_values[fold] = validate_fold_or_hold_out(net, valloader, loss_function, device)
        df_eval = sum(loss_values.values()) / len(loss_values.values())
        df_preds = np.nan
    except Exception as err:
        print(f"Error in Net_Core. Call to evaluate_cv() failed. {err=}, {type(err)=}")
        df_eval = np.nan
        df_preds = np.nan
    add_attributes(net, removed_attributes)
    return df_eval, df_preds


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
        if torch.cuda.is_available():
            device = "cuda:0"
            if torch.cuda.device_count() > 1:
                print("We will use", torch.cuda.device_count(), "GPUs!")
                net = nn.DataParallel(net)
        net.to(device)
        # loss_function = nn.CrossEntropyLoss()
        # TODO: optimizer = optim.Adam(net.parameters(), lr=lr)
        # optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
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
            print(f"Epoch: {epoch + 1}")
            # training loss from one epoch:
            _ = train_hold_out(
                net=net,
                trainloader=trainloader,
                batch_size=batch_size_instance,
                loss_function=loss_function,
                optimizer=optimizer,
                device=device,
                show_batch_interval=show_batch_interval,
            )
            # TODO: scheduler.step()
            # Early stopping check. Calculate validation loss from one epoch:
            val_accuracy, val_loss = validate_fold_or_hold_out(
                net, valloader=valloader, loss_function=loss_function, metric=metric, device=device
            )
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
    print(f"Returned to Spot: Validation loss: {df_eval}")
    print("----------------------------------------------")
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


def train_hold_out(net, trainloader, batch_size, loss_function, optimizer, device, show_batch_interval=10_000):
    running_loss = 0.0
    epoch_steps = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()
        epoch_steps += 1
        if i % show_batch_interval == (show_batch_interval - 1):  # print every show_batch_interval mini-batches
            print(
                "Batch: %5d. Batch Size: %d. Training Loss (running): %.3f"
                % (i + 1, int(batch_size), running_loss / epoch_steps)
            )
            running_loss = 0.0
    return loss.item()


def train_tuned(net, train_dataset, shuffle, loss_function, metric, device=None, show_batch_interval=10_000, path=None):
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
    )


def test_tuned(net, shuffle, test_dataset=None, loss_function=None, metric=None, device=None, path=None):
    batch_size_instance = net.batch_size
    removed_attributes, net = get_removed_attributes_and_base_net(net)
    if path is not None:
        net.load_state_dict(torch.load(path))
        net.eval()
    try:
        device = getDevice(device=device)
        if torch.cuda.is_available():
            device = "cuda:0"
            if torch.cuda.device_count() > 1:
                print("We will use", torch.cuda.device_count(), "GPUs!")
                net = nn.DataParallel(net)
        net.to(device)
        valloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=int(batch_size_instance), shuffle=shuffle, num_workers=0
        )
        metric_value, loss = validate_fold_or_hold_out(
            net, valloader=valloader, loss_function=loss_function, metric=metric, device=device
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
