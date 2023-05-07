import numpy as np
from sklearn.model_selection import KFold
import torch
from torch import nn
import torch.optim as optim
from spotPython.utils.device import getDevice
from torch.utils.data import random_split


class Net_Core(nn.Module):
    def __init__(self, lr, batch_size, epochs, k_folds):
        super(Net_Core, self).__init__()
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.k_folds = k_folds
        self.results = {}

    def reset_weights(self):
        for layer in self.children():
            if hasattr(layer, "reset_parameters"):
                print(f"Reset trainable parameters of layer = {layer}")
                layer.reset_parameters()

    def train_fold(self, trainloader, criterion, optimizer, device):
        for epoch in range(self.epochs):
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

    def validate_fold(self, valloader, criterion, device):
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1
        return 100.0 * (correct / total)

    def evaluate_cv(self, dataset, shuffle=False):
        try:
            device = getDevice()
            # if torch.cuda.device_count() > 1:
            #     self = nn.DataParallel(self)
            self.to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.parameters(), lr=self.lr)
            kfold = KFold(n_splits=self.k_folds, shuffle=shuffle)
            for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
                train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
                val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
                trainloader = torch.utils.data.DataLoader(
                    dataset, batch_size=self.batch_size, sampler=train_subsampler, num_workers=4
                )
                valloader = torch.utils.data.DataLoader(
                    dataset, batch_size=self.batch_size, sampler=val_subsampler, num_workers=4
                )
                self.reset_weights()
                self.train_fold(trainloader, criterion, optimizer, device)
                self.results[fold] = self.validate_fold(valloader, criterion, device)
            df_eval = sum(self.results.values()) / len(self.results.values())
            df_preds = np.nan
        except Exception as err:
            print(f"Error in Net_Core. Call to evaluate_cv() failed. {err=}, {type(err)=}")
            df_eval = np.nan
            df_preds = np.nan
        return df_eval, df_preds

    def evaluate_hold_out(self, dataset, shuffle, test_dataset=None):
        lr = self.lr
        epochs = self.epochs
        try:
            device = getDevice()
            self.to(device)
            criterion = nn.CrossEntropyLoss()
            # TODO: optimizer = optim.Adam(self.parameters(), lr=lr)
            optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
            if test_dataset is None:
                trainloader, valloader = self.create_train_val_data_loaders(dataset, shuffle)
            else:
                trainloader, valloader = self.create_train_test_data_loaders(dataset, shuffle, test_dataset)
            # TODO: scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
            # Early stopping parameters
            patience = 5
            best_val_loss = float("inf")
            counter = 0
            for epoch in range(epochs):
                self.train_hold_out(trainloader, criterion, optimizer, device=device, epoch=epoch)
                # TODO: scheduler.step()
                # Early stopping check
                val_accuracy, val_loss = self.validate_hold_out(valloader=valloader, criterion=criterion, device=device)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    counter = 0
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
        print(f"Returned to Spot: Validation loss: {df_eval}")
        return df_eval, df_preds

    def create_train_val_data_loaders(self, dataset, shuffle):
        test_abs = int(len(dataset) * 0.6)
        train_subset, val_subset = random_split(dataset, [test_abs, len(dataset) - test_abs])
        trainloader = torch.utils.data.DataLoader(
            train_subset, batch_size=int(self.batch_size), shuffle=shuffle, num_workers=8, pin_memory=True
        )
        valloader = torch.utils.data.DataLoader(
            val_subset, batch_size=int(self.batch_size), shuffle=shuffle, num_workers=8, pin_memory=True
        )
        return trainloader, valloader

    def create_train_test_data_loaders(self, dataset, shuffle, test_dataset):
        trainloader = torch.utils.data.DataLoader(
            dataset, batch_size=int(self.batch_size), shuffle=shuffle, num_workers=8, pin_memory=True
        )
        testloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=int(self.batch_size), shuffle=shuffle, num_workers=8, pin_memory=True
        )
        return trainloader, testloader

    def train_hold_out(self, trainloader, criterion, optimizer, device, epoch):
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = self(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 1000 == 999:  # print every 1000 mini-batches
                print(
                    "Epoch: %d, Batch: %5d. Batch Size: %d. Training Loss: %.3f"
                    % (epoch + 1, i + 1, int(self.batch_size), running_loss / epoch_steps)
                )
                running_loss = 0.0

    def validate_hold_out(self, valloader, criterion, device):
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        pred_list = []
        with torch.no_grad():
            for i, data in enumerate(valloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self(inputs)
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
