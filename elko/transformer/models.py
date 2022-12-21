import torch
import torch.nn as nn

class FeedForwardMLP(nn.Module):
    '''
    A Simple Feed Forward MLP for use after an attention block.

    ...

    Arguments
    ----------

    in_dims :int
        number of dimensions for the input x
    hidden_dims : int
        number of hidden states to expand to on forward pass
    dropout : float
        amount of dropout to apply after ReLU activation (0->1)

    TODO's
    ------

    * ReLU activation is hard coded
    * Not driven by config dict
    * Always assumes dropout exists (never will be set to 0)
    '''
    
    def __init__(self, 
                 in_dims:int, 
                 hidden_dims:int, 
                 dropout:float=0.1
                ):
        
        super(FeedForwardMLP, self).__init__()
        
        self.in_dims = in_dims
        self.hidden_dims = hidden_dims
        
        self.sequence = nn.Sequential(
            nn.Linear(self.in_dims, self.hidden_dims),
            nn.ReLU(),
            nn.Linear(self.hidden_dims, self.in_dims),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.sequence(x)



        