import megengine as mge
import numpy as np

import megengine.functional as F
import megengine.module as M
import megengine.optimizer as optim
from megengine.data import DataLoader, RandomSampler
from megengine.data.dataset import Dataset
from megengine.autodiff import GradManager


class XORDataset(Dataset):
    def __init__(self, X, Label):
        super().__init__()
        self.X = X
        self.Label = Label
        self.len = self.X.shape[0]
    
    def __getitem__(self, index):
        x = self.X[index, :]
        label = self.Label[index, :]
        return x, label
    
    def __len__(self):
        return self.len


class XORNet(M.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = M.Linear(2, 2)
        self.fc2 = M.Linear(2, 1)
    
    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.sigmoid(self.fc2(x))
        return x


bs = 2
lr = 0.05
epoch_num = 10000

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
Label = np.array([[0], [1], [1], [0]], dtype=np.float32)
dataset = XORDataset(X, Label)
dataloader = DataLoader(dataset, sampler=RandomSampler(dataset, batch_size=bs))

net = XORNet()
gm = GradManager().attach(net.parameters())
optimizer = optim.SGD(net.parameters(), lr=lr)

for epoch in range(epoch_num):
    if (epoch+1) % 100 == 0:
        print("Epoch: ", epoch)
    for x, label in dataloader:
        x = mge.Tensor(x)
        label = mge.Tensor(label)
        with gm:
            pred = net(x)
            loss = F.nn.square_loss(pred, label)
            gm.backward(loss)
            optimizer.step().clear_grad()
            if (epoch+1) % 100 == 0:
                print(x[0], " : ", pred[0])
                print(x[1], " : ", pred[1])

mge.save(net.state_dict(), "./xornet_state_dict.pkl")
