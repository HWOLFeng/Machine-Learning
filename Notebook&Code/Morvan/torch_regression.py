import torch
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.2*torch.rand(x.size())

# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()


# inherit
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)
        pass

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.predict(x)
        return x


net = Net(1, 10, 1)
# use optimizer optim the parameters of net
# the algorithm is SGD(Stochastic gradient descent) method
optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
loss_func = torch.nn.MSELoss()

# set iteration step is 100
for t in range(100):
    # output
    prediction = net(x)
    # calculate mean square error
    loss = loss_func(prediction, y)
    # init gradient
    optimizer.zero_grad()
    # BP
    loss.backward()
    # next step depend on learning rate
    optimizer.step()
    print(
        "\nprediction: ", prediction,
        "\nloss: ", loss
    )
