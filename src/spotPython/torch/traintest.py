import numpy as np
from sklearn.model_selection import KFold
import torch
from torch import nn
import torch.optim as optim
from spotPython.utils.device import getDevice
from torch.utils.data import random_split
from spotPython.utils.classes import get_additional_attributes


def remove_attributes(net, atttributes_to_remove):
    for attr in atttributes_to_remove:
        delattr(net, attr)
    return net


def reset_weights(net):
    for layer in net.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()


def train_fold(net, trainloader, epochs, criterion, optimizer, device, show_batch_interval=10_000):
    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}")
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % show_batch_interval == (show_batch_interval - 1):  # print every show_batch_interval mini-batches
                print("Batch: %5d. Training Loss (running): %.3f" % (i + 1, running_loss / epoch_steps))
                running_loss = 0.0


def validate_fold(net, valloader, criterion, device):
    val_loss = 0.0
    val_steps = 0
    total = 0
    correct = 0
    for i, data in enumerate(valloader, 0):
        with torch.no_grad():
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            val_loss += loss.cpu().numpy()
            val_steps += 1
    return 100.0 * (correct / total)


def evaluate_cv(
    net,
    dataset,
    shuffle=False,
    num_workers=0,
    device=None,
    lr=None,
    epochs=None,
    batch_size=None,
    k_folds=None,
    show_batch_interval=10_000,
):
    lr = net.lr
    epochs = net.epochs
    batch_size = net.batch_size
    k_folds = net.k_folds
    removed_attributes = get_additional_attributes(net)
    net = remove_attributes(net, removed_attributes)
    results = {}
    try:
        device = getDevice(device=device)
        if torch.cuda.is_available():
            device = "cuda:0"
            if torch.cuda.device_count() > 1:
                print("We will use", torch.cuda.device_count(), "GPUs!")
                net = nn.DataParallel(net)
        net.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=lr)
        kfold = KFold(n_splits=k_folds, shuffle=shuffle)
        for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
            print(f"Fold: {fold + 1}")
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
            trainloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, sampler=train_subsampler, num_workers=num_workers
            )
            valloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, sampler=val_subsampler, num_workers=num_workers
            )
            reset_weights(net)
            # Train fold for several epochs:
            train_fold(net, trainloader, epochs, criterion, optimizer, device, show_batch_interval=show_batch_interval)
            # Validate fold:
            results[fold] = validate_fold(net, valloader, criterion, device)
        df_eval = sum(results.values()) / len(results.values())
        df_preds = np.nan
    except Exception as err:
        print(f"Error in Net_Core. Call to evaluate_cv() failed. {err=}, {type(err)=}")
        df_eval = np.nan
        df_preds = np.nan
    net.lr = lr
    net.epochs = epochs
    net.batch_size = batch_size
    net.k_folds = k_folds
    return df_eval, df_preds


def evaluate_hold_out(
    net, train_dataset, shuffle, test_dataset=None, device=None, show_batch_interval=10_000, path=None, save_model=False
):
    lr = net.lr
    epochs = net.epochs
    batch_size = net.batch_size
    patience = net.patience
    removed_attributes = get_additional_attributes(net)
    net = remove_attributes(net, removed_attributes)
    try:
        device = getDevice(device=device)
        if torch.cuda.is_available():
            device = "cuda:0"
            if torch.cuda.device_count() > 1:
                print("We will use", torch.cuda.device_count(), "GPUs!")
                net = nn.DataParallel(net)
        net.to(device)
        criterion = nn.CrossEntropyLoss()
        # TODO: optimizer = optim.Adam(net.parameters(), lr=lr)
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
        if test_dataset is None:
            trainloader, valloader = create_train_val_data_loaders(
                dataset=train_dataset, batch_size=batch_size, shuffle=shuffle
            )
        else:
            trainloader, valloader = create_train_test_data_loaders(
                dataset=train_dataset, batch_size=batch_size, shuffle=shuffle, test_dataset=test_dataset
            )
        # TODO: scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        # Early stopping parameters
        patience = patience
        best_val_loss = float("inf")
        counter = 0
        # We only have "one fold" which is trained for several epochs
        # (we do not have to reset the weights for each fold):
        for epoch in range(epochs):
            print(f"Epoch: {epoch + 1}")
            # training loss from one epoch:
            _ = train_hold_out(
                net=net,
                trainloader=trainloader,
                batch_size=batch_size,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                show_batch_interval=show_batch_interval,
            )
            # TODO: scheduler.step()
            # Early stopping check. Calculate validation loss from one epoch:
            val_accuracy, val_loss = validate_hold_out(net, valloader=valloader, criterion=criterion, device=device)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                # save model:
                if save_model:
                    torch.save({"model_state_dict": net.state_dict()}, path)
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        df_eval = val_loss
        df_preds = np.nan
    except Exception as err:
        print(f"Error in Net_Core. Call to evaluate_hold_out() failed. {err=}, {type(err)=}")
        df_eval = np.nan
        df_preds = np.nan
    net.lr = lr
    net.epochs = epochs
    net.batch_size = batch_size
    net.patience = patience
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


def train_hold_out(net, trainloader, batch_size, criterion, optimizer, device, show_batch_interval=10_000):
    running_loss = 0.0
    epoch_steps = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        epoch_steps += 1
        if i % show_batch_interval == (show_batch_interval - 1):  # print every show_batch_interval mini-batches
            print(
                "Batch: %5d. Batch Size: %d. Training Loss (running): %.3f"
                % (i + 1, int(batch_size), running_loss / epoch_steps)
            )
            running_loss = 0.0
    return loss.item()


def validate_hold_out(net, valloader, criterion, device):
    val_loss = 0.0
    val_steps = 0
    total = 0
    correct = 0
    pred_list = []
    with torch.no_grad():
        for i, data in enumerate(valloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            pred_list.append(predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            val_loss += loss.cpu().numpy()
            val_steps += 1
    accuracy = correct / total
    loss = val_loss / val_steps
    print(f"Loss on hold-out set: {loss}")
    print(f"Accuracy on hold-out set: {accuracy}")
    return accuracy, loss


def train_save(net, train_dataset, shuffle, device=None, show_batch_interval=10_000, path=None, save_model=False):
    lr = net.lr
    epochs = net.epochs
    batch_size = net.batch_size
    patience = net.patience
    removed_attributes = get_additional_attributes(net)
    net = remove_attributes(net, removed_attributes)
    try:
        device = getDevice(device=device)
        if torch.cuda.is_available():
            device = "cuda:0"
            if torch.cuda.device_count() > 1:
                print("We will use", torch.cuda.device_count(), "GPUs!")
                net = nn.DataParallel(net)
        net.to(device)
        criterion = nn.CrossEntropyLoss()
        # TODO: optimizer = optim.Adam(net.parameters(), lr=lr)
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
        trainloader, valloader = create_train_val_data_loaders(
            dataset=train_dataset, batch_size=batch_size, shuffle=shuffle
        )
        # TODO: scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        # Early stopping parameters
        patience = patience
        best_val_loss = float("inf")
        counter = 0
        # We only have "one fold" which is trained for several epochs
        # (we do not have to reset the weights for each fold):
        for epoch in range(epochs):
            print(f"Epoch: {epoch + 1}")
            # training loss from one epoch:
            _ = train_hold_out(
                net=net,
                trainloader=trainloader,
                batch_size=batch_size,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                show_batch_interval=show_batch_interval,
            )
            # TODO: scheduler.step()
            # Early stopping check. Calculate validation loss from one epoch:
            val_accuracy, val_loss = validate_hold_out(net, valloader=valloader, criterion=criterion, device=device)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                # save model:
                if save_model:
                    torch.save({"model_state_dict": net.state_dict()}, path)
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        df_eval = val_loss
        df_preds = np.nan
    except Exception as err:
        print(f"Error in Net_Core. Call to evaluate_hold_out() failed. {err=}, {type(err)=}")
        df_eval = np.nan
        df_preds = np.nan
    net.lr = lr
    net.epochs = epochs
    net.batch_size = batch_size
    net.patience = patience
    print(f"Returned to Spot: Validation loss: {df_eval}")
    print("----------------------------------------------")
    return df_eval, df_preds


def test_saved(net, shuffle, test_dataset=None, device=None, show_batch_interval=10_000, path=None):
    lr = net.lr
    epochs = net.epochs
    batch_size = net.batch_size
    patience = net.patience
    removed_attributes = get_additional_attributes(net)
    net = remove_attributes(net, removed_attributes)
    try:
        device = getDevice(device=device)
        if torch.cuda.is_available():
            device = "cuda:0"
            if torch.cuda.device_count() > 1:
                print("We will use", torch.cuda.device_count(), "GPUs!")
                net = nn.DataParallel(net)
        net.to(device)
        criterion = nn.CrossEntropyLoss()
        valloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=int(batch_size), shuffle=shuffle, num_workers=0
        )
        val_accuracy, val_loss = validate_hold_out(net, valloader=valloader, criterion=criterion, device=device)
        df_eval = val_loss
        df_preds = np.nan
    except Exception as err:
        print(f"Error in Net_Core. Call to evaluate_hold_out() failed. {err=}, {type(err)=}")
        df_eval = np.nan
        df_preds = np.nan
    net.lr = lr
    net.epochs = epochs
    net.batch_size = batch_size
    net.patience = patience
    print(f"Returned to Spot: Validation loss: {df_eval}")
    print("----------------------------------------------")
    return df_eval, df_preds
