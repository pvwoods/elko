import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class ResnetBlock(nn.Module):

    def __init__(self, in_channels:int, out_channels:int, stride:int = 1, residual_layer:bool=True, conv_layer=None, conv_activation=None, out_activation=None) -> None:
        super(ResnetBlock, self).__init__()

        assert 0 < stride <= 2

        if conv_activation is None:
            conv_activation = nn.ReLU

        if conv_layer is None:
            conv_layer = nn.Conv2d

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.residual_layer = residual_layer

        self.sequence = nn.Sequential(
            conv_layer(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            conv_activation(),
            conv_layer(out_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels)
        )

        self.out_activation = nn.ReLU() if out_activation is None else out_activation()

    def identity(self, x):
        x = self.make_identity_matrix(x)
        if self.stride == 2:
            x = F.avg_pool2d(x, 2, 2)
        return x
    
    def make_identity_matrix(self, x):

        if self.in_channels == self.out_channels:
            return x

        # we need to add additional channels
        B, C, H, W = x.shape
        NC = self.out_channels - C
        z = torch.zeros(B, NC, H, W, dtype=x.dtype, device=x.device)

        return torch.cat([x, z], dim=1)

    def forward(self, x):

        out = self.sequence(x)
        if self.residual_layer:
            r = self.identity(x)
            print("b", r.shape, out.shape)
            out = self.out_activation(out + r)

        return out

class MinVAE(nn.Module):
    
    def __init__(self,
                 input_dims:int,
                 hidden_dims:int,
                 bottleneck_dims:int,
                 convolutions:List[Tuple[int,int,int]]=None, 
                 conv_activation=None, 
                 out_activation=None
                ):
        
        '''
        Minimal Variational Auto Encoder
        
        Attributes
        ----------
        
        input_dims : int 
            number of input channel dimensions
        hidden_dims : int
            number of channel dimensions for hidden layers
        bottleneck_dims : int
            channel dimensions at bottelneck
        convolutions : List[Tuple[int,int,int]] (optional)
            list of tuples for convolution layers, of the form (kernel size, stride, padding)
        conv_activation (optional)
            class of the activation to be used for actvations between convolutional layers.  Defaults to SiLU
        out_activation (optional)
            class of the activation to be used after final step of the deocder.  If not set, no activation will be used.
        '''
        
        super(MinVAE, self).__init__()

        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.bottleneck_dims = bottleneck_dims

        if convolutions is None:
            convolutions = [
                (4, 2, 1),
                (4, 2, 1),
                (5, 1, 0),
                (3, 1, 0)
            ]
            
        if conv_activation is None:
            conv_activation = nn.SiLU
        
        # build encoder
        
        encoder_steps = []
        for i, (k, s, p) in enumerate(convolutions):
            in_dim = input_dims if i == 0 else hidden_dims
            out_dim = (bottleneck_dims * 2) if i == len(convolutions) - 1 else hidden_dims
            encoder_steps.append(nn.Conv2d(in_dim, out_dim, k, s, p))
            encoder_steps.append(nn.BatchNorm2d(out_dim))
            # no activation on last layer
            if i != len(convolutions) - 1:
                encoder_steps.append(conv_activation())
        
        self.encoder = nn.Sequential(*encoder_steps)
        
        # build decoder
        
        decoder_steps = []
        for i, (k, s, p) in enumerate(convolutions[::-1]):
            in_dim = (bottleneck_dims * 2) if i == 0 else hidden_dims
            out_dim = input_dims if i == len(convolutions) - 1 else hidden_dims
            decoder_steps.append(nn.ConvTranspose2d(in_dim, out_dim, k, s, p))
            decoder_steps.append(nn.BatchNorm2d(out_dim))
            # no activation on last layer
            if i != len(convolutions) - 1:
                decoder_steps.append(conv_activation())
        
        self.decoder = nn.Sequential(*decoder_steps)
        
        if out_activation:
            self.out_activation = out_activation()
            
    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=1)
        
        h_dist = torch.distributions.Normal(mu, logvar.mul(.5).exp())
        reference_dist = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(logvar))
        kl_loss = torch.distributions.kl_divergence(h_dist, reference_dist).sum(dim=1).mean()
        
        out = self.decoder(h)
        if self.out_activation:
            out = self.out_activation(out)
        
        return out, kl_loss

class EmbeddingQuantizer(nn.Module):

    def __init__(self, codebook_size:int, embedding_dims:int) -> None:
        
        super(EmbeddingQuantizer, self).__init__()

        self.embedding_dims = embedding_dims
        self.codebook_size = codebook_size

        self.embeddings = nn.Embedding(self.codebook_size, self.embedding_dims)

    def forward(self, x):

        B, C, H, W = x.shape

        reshape_inputs = x.permute(0, 2, 3, 1).contiguous() # embed by channel values for each pixel (BHWC)
        reshape_inputs = reshape_inputs.view(-1, self.embedding_dims) # reshape to embedding dimensions (B, E)

        # calculate distances between all inputs and embeddings
        xs = (reshape_inputs**2).sum(dim=1, keepdim=True)
        ys = (self.embeddings.weight**2).sum(dim=1)
        dots = reshape_inputs @ self.embeddings.weight.t()
        distances = (xs + ys) - (2 * dots)

        # get embedding indices and quantize
        embedding_indexes = torch.argmin(distances, dim=1).unsqueeze(1)
        quantized_embeddings = self.embeddings(embedding_indexes).view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        loss = F.mse_loss(x, quantized_embeddings)

        return quantized_embeddings, embedding_indexes.squeeze(-1), loss

class MinVQVAE(MinVAE):
    
    def __init__(self,
                 input_dims:int,
                 hidden_dims:int,
                 bottleneck_dims:int,
                 codebook_size:int,
                 convolutions:List[Tuple[int,int,int]]=None, 
                 conv_activation=None, 
                 out_activation=None
                ):
        
        '''
        Minimal Vector Quantized Variational Auto Encoder
        
        Attributes
        ----------
        
        input_dims : int 
            number of input channel dimensions
        hidden_dims : int
            number of channel dimensions for hidden layers
        bottleneck_dims : int
            channel dimensions at bottelneck (also used as the embedding size for the codebook)
        codebook_size : int
            number of embeddings stored in the codebook
        convolutions : List[Tuple[int,int,int]] (optional)
            list of tuples for convolution layers, of the form (kernel size, stride, padding)
        conv_activation (optional)
            class of the activation to be used for actvations between convolutional layers.  Defaults to SiLU
        out_activation (optional)
            class of the activation to be used after final step of the deocder.  If not set, no activation will be used.
        '''
        
        super(MinVQVAE, self).__init__(
            input_dims, hidden_dims, bottleneck_dims,
            convolutions, conv_activation, out_activation)

        self.codebook_size = codebook_size
        self.codebook_dims = self.bottleneck_dims * 2 # 1 scalar for mean, 1 for std

        self.embedding_quantizer = EmbeddingQuantizer(self.codebook_size, self.codebook_dims)

    def forward(self, x):

        # run through encoder
        h = self.encoder(x)

        # now get quantized embeddings
        qh, codes, q_loss = self.embedding_quantizer(h)

        # chunk
        mu, logvar = qh.chunk(2, dim=1)
        
        h_dist = torch.distributions.Normal(mu, logvar.mul(.5).exp())
        reference_dist = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(logvar))
        kl_loss = torch.distributions.kl_divergence(h_dist, reference_dist).sum(dim=1).mean()
        
        out = self.decoder(h)
        if self.out_activation:
            out = self.out_activation(out)
        
        return out, codes, q_loss, kl_loss