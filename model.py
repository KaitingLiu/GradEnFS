# import lib
import torch
import torch.nn as nn

# MLP network
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.fc4 = nn.Linear(hidden_dim, output_dim, bias=True)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        # regist hooks to track the gradient flow
        self.relu.register_full_backward_hook(self.hook_fn)

        # save data
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.ac_grad = []

    def forward(self, x):
        h0 = x.view(-1, self.input_dim)
        h1 = self.relu(self.fc1(h0))
        h2 = self.relu(self.fc2(h1))
        h3 = self.relu(self.fc3(h2))
        y_hat = self.softmax(self.fc4(h3))
        return y_hat

    def hook_fn(self, module, grad_input, grad_output):
        self.ac_grad.append(torch.mean(torch.abs(grad_output[0]), dim=0))
