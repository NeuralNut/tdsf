import torch.nn as nn
import torch.nn.functional as F

# Fully Connected NN
class FCNN(nn.Module):

    def __init__(self, n_chan_in, n_chan_out, n_hidden, activation):
        super(FCNN, self).__init__()
        
        self.act = activation
        self.layers = nn.ModuleList([nn.Linear(n_chan_in, n_hidden[0])]) 
        self.layers.extend([nn.Linear(n_hidden[i],n_hidden[i+1]) for i in range(len(n_hidden)-1)])
        self.out = nn.Linear(n_hidden[-1], n_chan_out)
        
        
        
    def forward(self, x):
        for layer in self.layers:
            x = self.act(layer(x))
        return self.out(x)  