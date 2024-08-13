import torch
from torch import Tensor
import torch.nn as nn
import bayesflow as bf
from bayesflow.utils import filter_concatenate

class DeepAdaptiveDesign(nn.Module):
  def __init__(
      self,
      encoder_net: nn.Module | bf.networks.DeepSet, # same summary for bf and dad or different?
      decoder_net: nn.Module,
      design_shape: torch.Size, # [xi_dim]
      summary_variables: list[str] = None # in case of using summary_net from bf
    ) -> None:
    super().__init__()
    self.design_shape = design_shape
    self.register_parameter(
        "initial_design",
        nn.Parameter(0.1 * torch.ones(design_shape, dtype=torch.float32)) # scalar
    )
    self.encoder_net = encoder_net
    self.decoder_net = decoder_net
    self.summary_variables = summary_variables

  def forward(self, history, batch_size: int) -> Tensor:

    if history is None:
      return self.initial_design
    else:
      # embed design-outcome pairs
      embeddings = self.encoder_net(filter_concatenate(history, keys=self.summary_variables)).to('cpu').requires_grad_(True)  # in case of using summary_net from bf. [B, summary_dim]
      # get next design
      next_design = self.decoder_net(embeddings)
    return next_design

class EmitterNetwork(nn.Module):
  def __init__(
        self,
        input_dim, # summary_dim
        hidden_dim,
        output_dim, # xi_dim
        n_hidden_layers=2,
        activation=nn.Softplus,
    ):
    super().__init__()
    self.activation_layer = activation()
    self.input_layer = nn.Linear(input_dim, hidden_dim)
    if n_hidden_layers > 1:
      self.middle = nn.Sequential(
         *[
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), activation())
            for _ in range(n_hidden_layers - 1)
          ]
            )
    else:
      self.middle = nn.Identity()
      
    self.output_layer = nn.Linear(hidden_dim, output_dim)

  def forward(self, r):
    x = self.input_layer(r)
    x = self.activation_layer(x)
    x = self.middle(x)
    x = self.output_layer(x)
    return x.unsqueeze(1) # [B, xi_dim] -> [B, 1, xi_dim]
    
class RandomDesign(nn.Module):
    def __init__(self, design_shape: torch.Size):
        super().__init__()
        self.design_shape = design_shape

    def forward(self, history: dict, batch_size: int) -> Tensor:
        return torch.rand([batch_size, 1, self.design_shape[0]]) # [B, 1, xi_dim]