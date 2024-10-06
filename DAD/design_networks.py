import torch
from torch import Tensor
import torch.nn as nn
import bayesflow as bf
# from bayesflow.utils import filter_concatenate
import torch.nn.functional as F

# class DeepAdaptiveDesignSimple(nn.Module):
#     def __init__(
#         self,
#         design_size: int,
#         y_dim: int,
#         hidden_dim: int,
#         embedding_dim: int,
#         batch_size: int,
#     ):
#         super().__init__()
#         self.design_size = design_size
#         self.embedding = nn.Linear(design_size + y_dim, embedding_dim)
#         self.hidden_layer = nn.Linear(embedding_dim, hidden_dim)
#         self.output_layer = nn.Linear(hidden_dim, design_size)
#         self.batch_size = batch_size

#     def forward(self, history = None, batch_size: int = None) -> Tensor:
        
#         if history is None: # initial design
#            designs = torch.empty(0, 1, self.design_size)
#            outcomes = torch.empty(0, 1, 1)

#         else:
#            outcomes = history["outcomes"].transpose(1, 0); designs = history["designs"].transpose(1, 0)

#         inputs = torch.cat([designs, outcomes], dim=-1)  # [T, B, D + Y]
#         # encoder part: embed -> relu -> sum across T
#         embedded = F.relu(self.embedding(inputs))
#         summed = embedded.sum(dim=0)  # sum across T
#         # "decoder" part: relu -> hidden -> output
#         hidden = F.relu(self.hidden_layer(summed))
        
#         return torch.tanh(self.output_layer(hidden)).unsqueeze(-1)
    
# Network([embedding_dim, 4, design_size])

class DADSimple(nn.Module):
    def __init__(
        self,
        design_size: int,
        y_dim: int,
        embedding_dim: int,
        batch_size: int,
        encoder_net: nn.Module = None,
        emitter: nn.Module = None
    ) -> None:
        super().__init__()
        self.design_size = design_size
        self.encoder_net = nn.Sequential([nn.Linear(design_size + y_dim, embedding_dim), nn.ReLU()]) if encoder_net is None else encoder_net
        self.emitter =  Network([embedding_dim, 4, design_size]) if emitter is None else emitter
        self.batch_size = batch_size

    def forward(self, history = None, batch_size: int = None) -> Tensor:
        
        if history is None: # initial design
           designs = torch.empty(0, 1, self.design_size)
           outcomes = torch.empty(0, 1, 1)

        else:
           outcomes = history["outcomes"].transpose(1, 0); designs = history["designs"].transpose(1, 0); 
           
        inputs = torch.cat([designs, outcomes], dim=-1)  # [T, B, D + Y]
        x = self.encoder_net(inputs)
        x = x.sum(dim=0)  # sum across T
        x = self.emitter(x)
        out = torch.clamp(x, -1, 1)
        
        return out.unsqueeze(-1)

class DADMulti(nn.Module):
    def __init__(
        self,
        design_size: int,
        y_dim: int,
        context_dim: int,
        embedding_dim: int,
        batch_size: int
    ):
        super().__init__()
        self.design_size = design_size
        self.embedding = nn.Linear(design_size + y_dim, embedding_dim) # embedding_dim + context_dim
        self.emitter = Network([embedding_dim + context_dim, 4, design_size])
        self.batch_size = batch_size
        self.context_dim = context_dim
        self.embedding_dim = embedding_dim

    def forward(self, history = None, batch_size: int = None) -> Tensor:
        
        if history is None: # initial design
           x = torch.empty(1, self.embedding_dim + self.context_dim)

        else:
           outcomes = history["outcomes"].transpose(1, 0); designs = history["designs"].transpose(1, 0); # masks = history["masks"]; n_obs = history["n_obs"]
           inputs = torch.cat([designs, outcomes], dim=-1)  # [T, B, D + Y]
           x = F.relu(self.embedding(inputs))
           x = x.sum(dim=0)  # sum across T
           x = torch.concat([x, history["masks"]], dim = -1)
        
        x = self.emitter(x)
        out = torch.clamp(x, -1, 1)
        
        return out.unsqueeze(-1)

class DeepAdaptiveDesign(nn.Module):
  def __init__(
      self,
      encoder_net: nn.Module | bf.networks.DeepSet, # same summary for bf and dad or different?
      decoder_net: nn.Module,
      design_size: int, # [xi_dim]
      summary_variables: list[str] = None # in case of using summary_net from bf
    ) -> None:
    super().__init__()
    self.design_size = design_size
    self.register_parameter(
        "initial_design",
        nn.Parameter(0.1 * torch.ones(torch.Size([design_size]), dtype=torch.float32)) 
    )
    self.encoder_net = encoder_net
    self.decoder_net = decoder_net
    self.summary_variables = summary_variables

  def forward(self, history, batch_size: int) -> Tensor:

    if history is None:
      return self.initial_design
    else:
      # embed design-outcome pairs
      # if isinstance(self.encoder_net, bf.networks.DeepSet):
      #    embeddings = self.encoder_net(filter_concatenate(history, keys=self.summary_variables)).to('cpu')  # in case of using summary_net from bf. [B, summary_dim]
      # else:
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
        activation=nn.ReLU,
        activation_output=nn.Tanh,
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
                                      activation_output())

  def forward(self, r):
    x = self.input_layer(r)
    x = self.activation_layer(x)
    x = self.middle(x)
    x = self.output_layer(x)
    return x.unsqueeze(1) # [B, xi_dim] -> [B, 1, xi_dim]
    
class RandomDesign(nn.Module):
    def __init__(self, design_size: int):
        super().__init__()
        self.design_size = design_size

    def forward(self, history: dict, batch_size: int) -> Tensor:
        
        return torch.rand([batch_size, 1, self.design_size]) # [B, 1, xi_dim]
    

class Network(nn.Module):
    def __init__(self, layer_sizes):
        super(Network, self).__init__()
        self.layer_sizes = layer_sizes
        self.model = self.build_network()

    def build_network(self):
        layers = [
            layer
            for i, (in_size, out_size) in enumerate(zip(self.layer_sizes[:-1], self.layer_sizes[1:]))
            for layer in [nn.Linear(in_size, out_size)] + [nn.ReLU() if i < len(self.layer_sizes) - 2 else nn.Identity()]
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
# class ScalingLayer(nn.Module):
#     def __init__(self, min: float, max: float):
#         super().__init__()
#         self.min = min
#         self.max = max

#     def forward(self, x):
#         return self.min + ((x + 1) / 2) * (self.max - self.min)


