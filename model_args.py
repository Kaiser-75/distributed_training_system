from dataclasses import dataclass, field
from loguru import logger
from torch import nn
from torchfeather.model.moe import MoEArgs

@dataclass

class DeepSeekV3ModelArgs:
    max_seq_len: int = 4096*4
    vocab_size: int = 102400
    dim: int = 2048
    inter_dim: int = 10944
    moe_inter_dim: int = 1408
    n_layers: int = 27
    n_dense_layers: int = 1
    n_heads: int = 16
    norm_eps: float = 1e-5 

    moe_args: MoEArgs = field(default_factory=MoEArgs)

    #Multi-head latent attention (MLA)
    

