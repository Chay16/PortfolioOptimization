import torch

class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = 1

        self.rnn = torch.nn.RNN(self.input_size, self.hidden_size, self.n_layers)
        self.fc = torch.nn.Linear(self.hidden_size, self.output_size)
        
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        out, hidden = self.rnn(x, h0)
        out = out.contiguous().view(-1, self.hidden_size)
        out = self.fc(out)
        
        return out, hidden

class PSN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PSN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc = torch.nn.Linear(self.input_size, self.hidden_size)
        
    def forward(self, x):
        x = self.fc(x)
        x = torch.sum(x, axis=1)
        x = torch.sigmoid(x)
        
        return x


class NN:
    def __init__(self, NNtype, input_size, hidden_size, output_size):
        self.NNtype = NNtype
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.model = None
        
    def init_model(self):
        if self.NNtype == "MLP":
            self.model = torch.nn.Sequential(
                                                torch.nn.Linear(self.input_size, self.hidden_size),
                                                torch.nn.Sigmoid(),
                                                torch.nn.Linear(self.hidden_size, self.output_size)
                                            )
            print(self.model)
        elif self.NNtype == "RNN":
            self.model = RNN(self.input_size, self.hidden_size, self.output_size)
            print(self.model)
        elif self.NNtype == "PSN":
            self.model = PSN(self.input_size, self.hidden_size, self.output_size)
            print(self.model)
        else:
            return "NN Type not implemented. Choose between ['MLP', 'RNN', 'PSN']"