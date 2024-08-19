from torch import nn
import spotpython.torch.netcore as netcore


class Net_vbdp(netcore.Net_Core):
    def __init__(self, _L0, l1, dropout_prob, lr_mult, batch_size, epochs, k_folds, patience, optimizer, sgd_momentum):
        super(Net_vbdp, self).__init__(
            lr_mult=lr_mult,
            batch_size=batch_size,
            epochs=epochs,
            k_folds=k_folds,
            patience=patience,
            optimizer=optimizer,
            sgd_momentum=sgd_momentum,
        )
        # min 160 (= 2*2*2*20) neurons in first layer
        # min 80 (= 2*2*2*10) neurons in second layer
        # min 40 (= 2*2*2*5) neurons in third layer
        # min 20 (= 2*2*2*2*2) neurons in fourth layer
        l2 = l1 // 2
        l3 = l2 // 2
        l4 = l3 // 2
        # self.fc1 = nn.Linear(6112, l1)
        # self.fc1 = nn.Linear(196, l1)
        self.fc1 = nn.Linear(_L0, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, l3)
        self.fc4 = nn.Linear(l3, l4)
        self.fc5 = nn.Linear(l4, 11)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.dropout2 = nn.Dropout(p=dropout_prob / 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        x = self.softmax(x)
        return x
