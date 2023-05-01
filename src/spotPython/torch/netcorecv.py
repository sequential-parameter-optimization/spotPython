# import os
import numpy as np

from sklearn.model_selection import KFold
import torch
from torch import nn
import torch.optim as optim

# from torch.utils.data import random_split, DataLoader, ConcatDataset
# from torchvision import transforms


class Net_Core_CV(nn.Module):
    def __init__(self, lr, batch_size, epochs, k_folds):
        super(Net_Core_CV, self).__init__()
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.k_folds = k_folds
        self.results = {}
        if torch.cuda.device_count() > 1:
            print("We will use", torch.cuda.device_count(), "GPUs!")
        self = nn.DataParallel(self)

    def evaluate(self, fun_control):
        try:
            # device = "cpu"
            # if torch.cuda.is_available():
            #     device = "cuda:0"
            #     if torch.cuda.device_count() > 1:
            #         net = nn.DataParallel(net)
            # Get cpu, gpu or mps device for training.
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            print(f"Using {device} device")
            self.to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
            # TODO:
            # if checkpoint_dir:
            #     model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
            #     model.load_state_dict(model_state)
            #     optimizer.load_state_dict(optimizer_state)

            # TODO:
            # trainset, testset = load_data(data_dir)

            trainset = fun_control["train"]
            kfold = KFold(n_splits=self.k_folds, shuffle=True)

            # test_abs = int(len(trainset) * 0.6)
            # train_subset, val_subset = random_split(trainset, [test_abs, len(trainset) - test_abs])
            for fold, (train_ids, val_ids) in enumerate(kfold.split(trainset)):
                print(f"Fold {fold}")
                # Sample elements randomly from a given list of ids, no replacement.
                train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
                val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
                # Define data loaders for training and testing data in this fold
                trainloader = torch.utils.data.DataLoader(
                    trainset, batch_size=self.batch_size, sampler=train_subsampler
                )
                valloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, sampler=val_subsampler)
                self.reset_weights()
                for epoch in range(self.epochs):  # loop over the dataset multiple times
                    running_loss = 0.0
                    epoch_steps = 0
                    for i, data in enumerate(trainloader, 0):
                        # get the inputs; data is a list of [inputs, labels]
                        inputs, labels = data
                        inputs, labels = inputs.to(device), labels.to(device)

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward + backward + optimize
                        outputs = self(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()

                        # print statistics
                        running_loss += loss.item()
                        epoch_steps += 1
                        if i % 2000 == 1999:  # print every 2000 mini-batches
                            print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / epoch_steps))
                            running_loss = 0.0

                # Validation loss
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
                # Print accuracy
                print("Accuracy for fold %d: %d %%" % (fold, 100.0 * correct / total))
                print("--------------------------------")
                self.results[fold] = 100.0 * (correct / total)
                # TODO:
                # with tune.checkpoint_dir(epoch) as checkpoint_dir:
                # path = os.path.join(checkpoint_dir, "checkpoint")
                # torch.save((self.state_dict(), optimizer.state_dict()), path)
            # Print fold results
            print(f"k-fold CV results for {self.k_folds} folds")
            print("--------------------------------")
            sum = 0.0
            for key, value in self.results.items():
                print(f"Fold {key}: {value} %")
                sum += value
            avg = sum / len(self.results.items())
            print(f"Average: {avg} %")
            df_eval = avg
            df_preds = np.nan
        except Exception as err:
            print(f"Error in Net_Core. Call to evaluate() failed. {err=}, {type(err)=}")
            df_eval = np.nan
            df_preds = np.nan
        return df_eval, df_preds

    def reset_weights(self):
        for layer in self.children():
            if hasattr(layer, "reset_parameters"):
                print(f"Reset trainable parameters of layer = {layer}")
                layer.reset_parameters()

    def test_accuracy(self, fun_control):
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using {device} device")
        self.to(device)

        # trainset, testset = load_data()
        testset = fun_control["test"]

        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, shuffle=False, num_workers=2)

        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct / total
