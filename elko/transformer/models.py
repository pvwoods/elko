import torch
import torch.nn as nn
import math


class FeedForwardMLP(nn.Module):
    """
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
    """

    def __init__(self, in_dims: int, hidden_dims: int, dropout: float = 0.1):

        super(FeedForwardMLP, self).__init__()

        self.in_dims = in_dims
        self.hidden_dims = hidden_dims

        self.sequence = nn.Sequential(
            nn.Linear(self.in_dims, self.hidden_dims),
            nn.ReLU(),
            nn.Linear(self.hidden_dims, self.in_dims),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.sequence(x)


class CausalSelfAttention(nn.Module):
    def __init__(
        self, embedding_dims: int, heads: int, block_size: int, dropout: float = 0.1
    ):

        """
        Performs multi-headed, causal self attention over input embeddings. Assumes input
        tensor of shape batch, tokens, embeddings (B, T, E) and returns a tensor of the
        same shape.

        ...


        Arguments
        ---------

        embedding_dims : int
            size of the embedding dimension
        heads : int
            number of heads in the attention block
        block_size : int
            maximum size of an input (T dimension)
        dropout : float
            amount of dropout to apply after attention. 0->1

        TODO's
        ------
        * not config driven
        * dropout always assumed to exist
        """

        super(CausalSelfAttention, self).__init__()

        assert (
            embedding_dims % heads == 0
        )  # we can't have leftover embedding dims after splitting into heads

        self.heads = heads
        self.embedding_dims = embedding_dims
        self.block_size = block_size
        self.head_dims = self.embedding_dims // self.heads  # head dims, D
        self.dk_sqrt = math.sqrt(self.head_dims)  # normalization term for attention

        # (3 * E) since we need (q,k,v) each of size E
        self.qkv_projection = nn.Linear(embedding_dims, 3 * embedding_dims)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_projection = nn.Linear(embedding_dims, embedding_dims)
        self.residual_dropout = nn.Dropout(dropout)

        # mask to apply to ensure we only look leftward when applying attention
        mask = torch.tril(torch.ones(self.block_size, self.block_size))[None, None, ...]
        self.register_buffer("mask", mask)

    def calculate_attention(self, q, k, v):
        """
        calculates Scaled Dot-Product Attention:

            Attention(Q, K, V) = softmax(QKᵀ/√dₖ)V

        ...


        Arguments
        ---------

        q : Tensor
            query tensor
        k : Tensor
            key tensor
        v : Tensor
            value tensor
        """

        (
            B,
            H,
            T,
            D,
        ) = q.shape  # Batch, Head, Token, Head Dims (assumes q,k,v all same shape)

        # calculate dot product similarity between queries and keys (B, H, T, D) @ (B, H, D, T) = (B, H, T, T)
        qk_similarity = q @ k.transpose(-2, -1) / self.dk_sqrt

        # mask tokens to the left of the current position
        masked_similarity = qk_similarity.masked_fill(
            self.mask[:, :, :T, :T] == 0, float("-inf")
        )

        # softmax on last dimension
        attention = torch.softmax(masked_similarity, dim=-1)

        # apply dropout
        attention = self.attn_dropout(attention)

        # apply attention to values (B, H, T, T) @ (B, H, T, D) = (B, H, T, D)
        out = attention @ v

        return out

    def forward(self, x):

        B, T, E = x.shape  # Batch size, Tokens, Embedding Dims (B, T, E)

        # forward projection and split into q, k, v
        qkv = self.qkv_projection(x).split(self.embedding_dims, dim=-1)

        # reshape to (B, T, H, D), then swap T and H dims so it's ordered by head, not token (B, H, T, D)
        q, k, v = [
            c.view(B, T, self.heads, self.head_dims).transpose(1, 2) for c in qkv
        ]

        # calculate attention
        attention = self.calculate_attention(q, k, v)

        # swap back H and T to get (B, T, H, D), then concat all heads
        attention = attention.transpose(1, 2).contiguous().view(B, T, E)

        # feed through outward projection
        out = self.out_projection(attention)

        # return with residual dropout
        return self.residual_dropout(out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_dims: int,
        heads: int,
        block_size: int,
        ff_dims: int,
        dropout: float = 0.1,
    ):

        """
        Basic transformer block.  Applies attention followed by feed forward with residual connections.

        ...

        Arguments
        ---------

        embedding_dims : int
            size of the embedding dimension
        heads : int
            number of heads in the attention block
        block_size : int
            maximum size of an input (T dimension)
        ff_dims : int
            number of hidden states for the feed forward layers
        dropout : float
            amount of dropout to apply after attention. 0->1

        TODO's
        ------
        * not config driven
        * dropout always assumed to exist
        """

        super(TransformerBlock, self).__init__()

        self.embedding_dims = embedding_dims
        self.heads = heads
        self.block_size = block_size
        self.ff_dims = ff_dims
        self.dropout = dropout

        self.attention_block = CausalSelfAttention(
            self.embedding_dims, self.heads, self.block_size, self.dropout
        )
        self.attention_norm = nn.LayerNorm(self.embedding_dims)
        self.feed_forward = FeedForwardMLP(self.embedding_dims, self.ff_dims)
        self.feed_forward_norm = nn.LayerNorm(self.embedding_dims)

    def forward(self, x):
        x = x + self.attention_block(self.attention_norm(x))
        x = x + self.feed_forward(self.feed_forward_norm(x))
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, embedding_dims: int, block_size: int):

        """
        Helper module to add positional embedding to an input
        embedding

        Arguments
        ---------

        embedding_dims : int
            size of the embedding dimension
        block_size : int
            maximum size of an input (T dimension)
        """

        super(PositionalEmbedding, self).__init__()

        self.embedding_dims = embedding_dims
        self.block_size = block_size

        self.embedding = nn.Embedding(block_size, embedding_dims)

    def forward(self, x):

        B, T, E = x.shape  # batch size, tokens, embedding dims

        # create range for size of this batch
        position_range = torch.arange(T).to(x.device)
        # get embeddings, then add Batch broadcasting dimension
        position_embeddings = self.embedding(position_range)[None, ...]

        x = x + position_embeddings

        return x


class CausalTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        embedding_dims: int,
        heads: int,
        block_size: int,
        ff_dims: int,
        dropout: float = 0.1,
    ):

        """
        A GPT style, decoder only causal transformer

        ...

        Arguments
        ---------

        vocab_size : int
            total number of valid tokens in the tokenizer vocabulary
        num_layers : int
            number of transformer block layers to stack for the forward pass after the embedding step
        embedding_dims : int
            size of the embedding dimension
        heads : int
            number of heads in the attention block
        block_size : int
            maximum size of an input (T dimension)
        ff_dims : int
            number of hidden states for the feed forward layers
        dropout : float
            amount of dropout to apply after attention. 0->1

        TODO's
        ------
        * not config driven
        * dropout always assumed to exist
        """

        super(CausalTransformer, self).__init__()

        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.embedding_dims = embedding_dims
        self.heads = heads
        self.block_size = block_size
        self.ff_dims = ff_dims
        self.dropout = dropout

        self.token_embedding = nn.Embedding(self.vocab_size, self.embedding_dims)
        self.positional_embedding = PositionalEmbedding(
            self.embedding_dims, self.block_size
        )
        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(embedding_dims, heads, block_size, ff_dims, dropout)
                for _ in range(self.num_layers)
            ]
        )

        self.lm_head = nn.Linear(self.embedding_dims, self.vocab_size)

    def forward(self, x):

        tok_emb = self.token_embedding(x)
        pos_emb = self.positional_embedding(tok_emb)
        hidden_states = self.transformer_blocks(pos_emb)

        logits = self.lm_head(hidden_states)

        return logits
