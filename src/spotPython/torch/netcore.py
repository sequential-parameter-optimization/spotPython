from torch import nn
import torch.nn.functional as F
from spotPython.utils.file import load_data
import torch.optim as optim
import torch
import os
from torch.utils.data import random_split
import numpy as np


class Net_Core(nn.Module):
    def __init__(self, lr, batch_size, epochs):
        super(Net_Core, self).__init__()
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
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

            test_abs = int(len(trainset) * 0.6)
            train_subset, val_subset = random_split(trainset, [test_abs, len(trainset) - test_abs])

            trainloader = torch.utils.data.DataLoader(
                train_subset,
                batch_size=int(self.batch_size),
                shuffle=True,
                num_workers=8,
            )
            valloader = torch.utils.data.DataLoader(
                val_subset,
                batch_size=int(self.batch_size),
                shuffle=True,
                num_workers=8,
            )

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
                pred_list = []
                for i, data in enumerate(valloader, 0):
                    with torch.no_grad():
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

                # TODO:
                # with tune.checkpoint_dir(epoch) as checkpoint_dir:
                # path = os.path.join(checkpoint_dir, "checkpoint")
                # torch.save((self.state_dict(), optimizer.state_dict()), path)
            df_eval = val_loss / val_steps
            df_preds = pred_list
            accuracy = correct / total
            print(f"Accuracy of the network on the validation data: {accuracy}")
        except Exception as err:
            print(f"Error in Net_Core. Call to evaluate() failed. {err=}, {type(err)=}")
            df_eval = np.nan
            df_preds = np.nan
        return df_eval, df_preds

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
