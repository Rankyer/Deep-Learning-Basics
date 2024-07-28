import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return x

def train_network(network, data, target_values, epochs=1000, learning_rate=0.001):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)

    data = torch.tensor(data, dtype=torch.float32)
    target_values = torch.tensor(target_values, dtype=torch.float32).view(-1, 1)

    for epoch in range(epochs):
        network.train()
        optimizer.zero_grad()
        outputs = network(data)
        loss = criterion(outputs, target_values)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.3f}')

data = np.array([
    [-2, -1],
    [25, 6],
    [17, 4],
    [-15, -6],
])
target_values = np.array([1, 0, 0, 1])
network = NN()
train_network(network, data, target_values)

def predict(network, data):
    network.eval()
    with torch.no_grad():
        data = torch.tensor(data, dtype=torch.float32)
        predictions = network(data)
        return predictions.numpy()

sample1 = np.array([-7, -3])
sample2 = np.array([20, 2])
print("Emily: %.3f" % predict(network, sample1).item())
print("Frank: %.3f" % predict(network, sample2).item())
