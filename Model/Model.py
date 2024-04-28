import torch.nn as nn
from Config import DENSE1_SIZE, DENSE1_ACTIVATION, DENSE2_SIZE, DENSE2_ACTIVATION


def create_py_torch_model(input_shape):
    model = Model(input_shape)
    return model


class Model(nn.Module):
    def __init__(self, input_shape):
        super(Model, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_shape, DENSE1_SIZE)
        self.activation1 = getattr(nn, DENSE1_ACTIVATION)()
        self.fc2 = nn.Linear(DENSE1_SIZE, DENSE2_SIZE)
        self.activation2 = getattr(nn, DENSE2_ACTIVATION)()

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.activation1(x)
        x = self.fc2(x)
        x = self.activation2(x)
        return x
