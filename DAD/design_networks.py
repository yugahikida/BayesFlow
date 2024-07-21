import torch
from torch import Tensor
import torch.nn as nn
import bayesflow as bf


class DeepAdaptiveDesign(nn.Module):
  def __init__(
      self,
      encoder_net: nn.Module | bf.networks.DeepSet, # same summary for bf and dad or different?
      decoder_net: nn.Module,
      design_shape: torch.Size
    ) -> None:
    super().__init__()
    self.design_shape = design_shape
    # initialise first design with random normal
    self.register_parameter(
        "initial_design",
        nn.Parameter(0.1 * torch.ones(design_shape, dtype=torch.float32))
    )
    self.encoder_net = encoder_net
    self.decoder_net = decoder_net

  def forward(self, designs=list[Tensor], outcomes=list[Tensor]) -> Tensor:
    if len(outcomes) == 0:
      return self.initial_design
    else:
      # embed design-outcome pairs
      embeddings = self.encoder_net(designs, outcomes)
      # get next design
      next_design = self.decoder_net(embeddings)
    return next_design
  

  class EmitterNetwork(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
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
        return x
    
class RandomDesign(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch_size: torch.Size, designs: [Tensor] = None, outcomes: [Tensor] = None) -> Tensor:
        return torch.rand(batch_size)