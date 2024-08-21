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
      if isinstance(self.encoder_net, bf.networks.DeepSet):
         embeddings = self.encoder_net(filter_concatenate(history, keys=self.summary_variables)).to('cpu')  # in case of using summary_net from bf. [B, summary_dim]
      else:
        outcomes = history["outcomes"].transpose(1, 0); designs = history["designs"].transpose(1, 0) # [B, tau, y_dim] -> [tau, B, y_dim]
        embeddings = torch.stack([self.encoder_net(xi, y) for (xi, y) in zip(designs, outcomes)], dim = 0).sum(dim = 0) # [tau, B, summary_dim] -> [B, summary_dim] 
      # get next design
      next_design = self.decoder_net(embeddings)
    return next_design
  

class EncoderNetwork(nn.Module):
    def __init__(
        self,
        xi_dim,
        y_dim,
        hidden_dim,
        encoding_dim,
        n_hidden_layers=2,
        activation=nn.Softplus,
    ):
        super().__init__()
        self.encoding_dim = encoding_dim
        self.activation_layer = activation()
        self.input_layer = nn.Linear(xi_dim + y_dim, hidden_dim)
        if n_hidden_layers > 1:
            self.middle = nn.Sequential(
                *[
                    nn.Sequential(nn.Linear(hidden_dim, hidden_dim), activation())
                    for _ in range(n_hidden_layers - 1)
                ]
            )
        else:
            self.middle = nn.Identity()
        self.output_layer = nn.Linear(hidden_dim, encoding_dim)

    def forward(self, xi, y, **kwargs):
        inputs = torch.concat([xi, y], dim=-1) # [B, xi_dim + y_dim]
        x = self.input_layer(inputs)
        x = self.activation_layer(x)
        x = self.middle(x)
        x = self.output_layer(x)
        return x

class EmitterNetwork(nn.Module):
  def __init__(
        self,
        input_dim, # summary_dim
        hidden_dim,
        output_dim, # xi_dim
        n_hidden_layers=2,
        activation=nn.Softplus,
        activation_output=nn.Tanh,
        scaler = 5
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
      
    self.output_layer = nn.Sequential(nn.Linear(hidden_dim, output_dim),
                                      activation_output(), 
                                      ScalingLayer(scaler)) # restrict xi to [-5, 5]

  def forward(self, r):
    x = self.input_layer(r)
    x = self.activation_layer(x)
    x = self.middle(x)
    x = self.output_layer(x)
    return x.unsqueeze(1) # [B, xi_dim] -> [B, 1, xi_dim]
    
class RandomDesign(nn.Module):
    def __init__(self, design_shape: torch.Size, scaler:int = 5):
        super().__init__()
        self.design_shape = design_shape
        self.scaler = scaler

    def forward(self, history: dict, batch_size: int) -> Tensor:
        return torch.randint(-self.scaler, self.scaler, [batch_size, 1, self.design_shape[0]]) # [B, 1, xi_dim] (restrict xi to [-5, 5])
    
class ScalingLayer(nn.Module):
    def __init__(self, scale_factor):
        super(ScalingLayer, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return x * self.scale_factor
