import os
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

    def evaluate_cv(self, dataset, shuffle=False):
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

            # dataset = fun_control["train"]
            kfold = KFold(n_splits=self.k_folds, shuffle=shuffle)

            # test_abs = int(len(dataset) * 0.6)
            # train_subset, val_subset = random_split(dataset, [test_abs, len(dataset) - test_abs])
            for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
                print(f"Fold {fold}")
                # Sample elements randomly from a given list of ids, no replacement.
                train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
                val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
                # Define data loaders for training and testing data in this fold
                trainloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, sampler=train_subsampler)
                valloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, sampler=val_subsampler)
                self.reset_weights()
                # Define best_score, counter, and patience for early stopping:
                best_score = None
                counter = 0
                patience = 10
                # path = os.path.join(".", "checkpoint")
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
                # early stopping:
                # https://stackoverflow.com/questions/60200088/how-to-make-early-stopping-in-image-classification-pytorch
                if best_score is None:
                    best_score = val_loss
                else:
                    # Check if val_loss improves or not.
                    if val_loss < best_score:
                        # val_loss improves, we update the latest best_score,
                        # and save the current model
                        best_score = val_loss
                        # TODO:
                        # torch.save({'state_dict':self.state_dict()}, path)
                    else:
                        # val_loss does not improve, we increase the counter,
                        # stop training if it exceeds the amount of patience
                        counter += 1
                        if counter >= patience:
                            break
                # TODO:
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