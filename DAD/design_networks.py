import torch
from torch import Tensor
import torch.nn as nn
import bayesflow as bf
# from bayesflow.utils import filter_concatenate
import torch.nn.functional as F
import random

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
        batch_size: int,
        T: int,
        emitter_dim: int = 4,
    ):
        super().__init__()
        self.design_size = design_size
        self.embedding = nn.Linear(design_size + y_dim, embedding_dim) # embedding_dim + context_dim
        self.emitter = nn.Sequential(Network([embedding_dim + context_dim, emitter_dim]), nn.ReLU())
        self.batch_size = batch_size
        self.context_dim = context_dim
        self.embedding_dim = embedding_dim
        self.last_layer = nn.ModuleList(
            [   
                nn.Sequential(
                    nn.Linear(emitter_dim, 1)
                )
                for _ in range(1 + T // 3)
            ])

    def forward(self, history = None, batch_size: int = None) -> Tensor:
        
        if history is None: # initial design
           x = torch.empty(1, self.embedding_dim + self.context_dim); tau = 0

        
        else:
           outcomes = history["outcomes"].transpose(1, 0); designs = history["designs"].transpose(1, 0); masks = history["masks"]; tau = int((history["n_obs"] ** 2)[0].item())
           inputs = torch.cat([designs, outcomes], dim=-1)  # [T, B, D + Y]
           x = F.relu(self.embedding(inputs))
           x = x.sum(dim=0)  # sum across T
           x = torch.concat([x, masks], dim = -1)
           assert x.isnan().sum().item() == 0

        x = self.emitter(x)
        assert x.isnan().sum().item() == 0
        x = self.last_layer[(1 + tau) // 3](x) if tau != 0 else self.last_layer[0](x)
        # x = torch.nan_to_num(x)
        out = torch.clamp(x, -1, 1)
        # out = torch.tanh(x)
        return out.unsqueeze(-1)
    
class DADMulti2(nn.Module):
    def __init__(
        self,
        design_size: int,
        y_dim: int,
        context_dim: int,
        embedding_dim: int,
        batch_size: int,
        T: int,
        emitter_dim: int = 4,
    ):
        super().__init__()
        self.design_size = design_size
        self.embedding = nn.Linear(design_size + y_dim, embedding_dim) # embedding_dim + context_dim
        self.emitter = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embedding_dim + context_dim, 1)
                )
                for _ in range(T - 1 // 3)
            ])
        # self.emitter = Network([embedding_dim + context_dim, design_size])
        self.batch_size = batch_size
        self.context_dim = context_dim
        self.embedding_dim = embedding_dim
        self.initial_design = nn.Parameter(0.1 * torch.ones(torch.Size([design_size]), dtype=torch.float32))

    def forward(self, history = None, batch_size: int = None) -> Tensor:
        
        if history is None: # initial design
           # x = torch.empty(1, self.embedding_dim + self.context_dim); tau = 0
           # x = self.initial_design
           x = torch.empty(1, 1); tau = 0

        else:
           outcomes = history["outcomes"].transpose(1, 0); designs = history["designs"].transpose(1, 0); masks = history["masks"]; tau = int((history["n_obs"] ** 2)[0].item())
           inputs = torch.cat([designs, outcomes], dim=-1)  # [T, B, D + Y]
           x = F.relu(self.embedding(inputs))
           x = x.sum(dim=0)  # sum across T
           x = torch.concat([x, masks], dim = -1)
           # x = self.emitter(x)
           x = self.emitter[tau - 1 // 3](x)
        out = torch.clamp(x, -1, 1)
        # out = torch.tanh(x)
        # out = torch.nan_to_num(out)
        return out.unsqueeze(-1)
    

class DADMulti3(nn.Module):
    def __init__(
        self,
        design_size: int,
        y_dim: int,
        context_dim: int,
        embedding_dim: int,
        batch_size: int,
        T: int,
        emitter_dim: int = 4,
    ):
        super().__init__()
        self.design_size = design_size
        self.embedding = nn.Linear(design_size + y_dim, embedding_dim) # embedding_dim + context_dim
        self.emitter = Network([embedding_dim + context_dim + 1, design_size])
        self.batch_size = batch_size
        self.context_dim = context_dim
        self.embedding_dim = embedding_dim
        self.initial_design = nn.Parameter(0.1 * torch.ones(torch.Size([design_size]), dtype=torch.float32))

    def forward(self, history = None, batch_size: int = None) -> Tensor:
        
        if history is None: # initial design
           # x = torch.empty(1, self.embedding_dim + self.context_dim); tau = 0
           x = self.initial_design

        else:
           outcomes = history["outcomes"].transpose(1, 0); designs = history["designs"].transpose(1, 0); masks = history["masks"]; tau = history["n_obs"]
           inputs = torch.cat([designs, outcomes], dim=-1)  # [T, B, D + Y]
           x = F.relu(self.embedding(inputs))
           x = x.sum(dim=0)  # sum across T
           x = torch.concat([x, masks, tau], dim = -1)
        
           x = self.emitter(x)
        # x = torch.nan_to_num(x)
        out = torch.clamp(x, -1, 1)
        # out = torch.tanh(x)
        return out.unsqueeze(-1)
    
class DADMulti4(nn.Module):
    def __init__(
        self,
        design_size: int,
        y_dim: int,
        context_dim: int,
        embedding_dim: int,
        batch_size: int,
        T: int,
        emitter_dim: int = 4,
        freeze: bool = False
    ):
        super().__init__()
        self.design_size = design_size
        self.embedding = nn.Linear(design_size + y_dim, embedding_dim) # embedding_dim + context_dim
        # self.emitter = nn.Sequential(Network([embedding_dim + context_dim, emitter_dim]), nn.ReLU())
        self.batch_size = batch_size
        self.context_dim = context_dim
        self.embedding_dim = embedding_dim
        self.last_layer = nn.ModuleList(
            [   
                nn.Sequential(
                    nn.Linear(embedding_dim + context_dim, design_size)
                )
                for _ in range(1 + T // 3)
            ])
        self.freeze = freeze
    
    # def set_freeze(self, freeze: bool) -> None:
    #     self.freeze = freeze

    def forward(self, history = None, batch_size: int = None) -> Tensor:
        
        if not self.freeze:
            if history is None: # initial design
                x = torch.empty(1, self.embedding_dim + self.context_dim); tau = 0
            else:
                outcomes = history["outcomes"].transpose(1, 0); designs = history["designs"].transpose(1, 0); masks = history["masks"]; tau = int((history["n_obs"] ** 2)[0].item())
                inputs = torch.cat([designs, outcomes], dim=-1)  # [T, B, D + Y]
                x = F.relu(self.embedding(inputs))
                x = x.sum(dim=0)  # sum across T
                x = torch.concat([x, masks], dim = -1)
            x = self.last_layer[(1 + tau) // 3](x) if tau != 0 else self.last_layer[0](x)
        else:
           with torch.no_grad():
              if history is None: # initial design
                x = torch.empty(1, self.embedding_dim + self.context_dim); tau = 0
              else:
                outcomes = history["outcomes"].transpose(1, 0); designs = history["designs"].transpose(1, 0); masks = history["masks"]; tau = int((history["n_obs"] ** 2)[0].item())
                inputs = torch.cat([designs, outcomes], dim=-1)  # [T, B, D + Y]
                x = F.relu(self.embedding(inputs))
                x = x.sum(dim=0)  # sum across T
                x = torch.concat([x, masks], dim = -1)
           x = self.last_layer[(1 + tau) // 3](x) if tau != 0 else self.last_layer[0](x)
        # assert x.isnan().sum().item() == 0
        x = torch.nan_to_num(x)
        out = torch.clamp(x, -1, 1)
        # out = torch.tanh(x)
        return out.unsqueeze(-1)
    
class DADMulti5(nn.Module):
    """
    multiple last layers for different timesteps
    """
    def __init__(
        self,
        design_size: int,
        y_dim: int,
        context_dim: int,
        embedding_dim: int,
        batch_size: int,
        T: int,
        emitter_dim: int = 4,
    ):
        super().__init__()
        self.design_size = design_size
        self.embedding = nn.Linear(design_size + y_dim, embedding_dim) # embedding_dim + context_dim
        self.emitter = Network([embedding_dim + context_dim, 4])
        self.residual_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(4, 4), nn.ReLU()
                )
                for _ in range(2)
            ])
        self.last_layer = nn.Linear(4, 1)
        self.batch_size = batch_size
        self.context_dim = context_dim
        self.embedding_dim = embedding_dim
    def forward(self, history = None, batch_size: int = None) -> Tensor:

        nn_list = [0, 0, 0, 0, 1, 1, 1]
        
        if history is None:
           x = torch.empty(1, self.embedding_dim + self.context_dim); tau = 0

        else:
           outcomes = history["outcomes"].transpose(1, 0); designs = history["designs"].transpose(1, 0); masks = history["masks"]; tau = int((history["n_obs"] ** 2)[0].item())
           inputs = torch.cat([designs, outcomes], dim=-1)  # [T, B, D + Y]
           x = F.relu(self.embedding(inputs))
           x = x.sum(dim=0)  # sum across T
           x = torch.concat([x, masks], dim = -1)
           
        # x = self.emitter(x)
        x = self.emitter(x)
        if tau >= 3:
           x += self.residual_layers[nn_list[tau - 3]](x)
        x = self.last_layer(x)
    
        out = torch.clamp(x, -1, 1)
        return out.unsqueeze(-1)
    

class DADMulti6(nn.Module):
    """
    Single network for all tau
    """
    def __init__(
        self,
        design_size: int,
        y_dim: int,
        context_dim: int,
        embedding_dim: int,
        batch_size: int,
        T: int,
        emitter_dim: int = 4,
    ):
        super().__init__()
        self.design_size = design_size
        self.embedding = nn.Linear(design_size + y_dim, embedding_dim) # embedding_dim + context_dim
        self.emitter = Network([embedding_dim + context_dim, 1])
        self.batch_size = batch_size
        self.context_dim = context_dim
        self.embedding_dim = embedding_dim
        self.initial_design = nn.Parameter(torch.randn(design_size))

    def forward(self, history = None, batch_size: int = None) -> Tensor:
        
        if history is None:
           x = self.initial_design; tau = 0
        else:
           outcomes = history["outcomes"].transpose(1, 0); designs = history["designs"].transpose(1, 0); masks = history["masks"]; tau = int((history["n_obs"] ** 2)[0].item())
           inputs = torch.cat([designs, outcomes], dim=-1)  # [T, B, D + Y]
           x = F.relu(self.embedding(inputs))
           x = x.sum(dim=0)  # sum across T
           x = torch.concat([x, masks], dim = -1)
           x = self.emitter(x)
    
        out = torch.clamp(x, -1, 1)
        return out.unsqueeze(-1)
    
class DADMulti7(nn.Module):
    """
    different last layers for all different timesteps THISSSSSS!!!
    """
    def __init__(
        self,
        design_size: int,
        y_dim: int,
        context_dim: int,
        embedding_dim: int,
        batch_size: int,
        T: int,
        hidden_dim: int = 4,
    ):
        super().__init__()
        self.design_size = design_size
        self.embedding = nn.Linear(design_size + y_dim, embedding_dim) # embedding_dim + context_dim
        self.emitter = nn.Sequential(Network([embedding_dim + context_dim, hidden_dim]), nn.ReLU())
        self.last_layer = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, 1)
                )
                for _ in range(T)
            ])
        self.batch_size = batch_size
        self.context_dim = context_dim
        self.embedding_dim = embedding_dim
    def forward(self, history = None, batch_size: int = None) -> Tensor:
        
        if history is None:
           x = torch.empty(1, self.embedding_dim + self.context_dim); tau = 0

        else:
           outcomes = history["outcomes"].transpose(1, 0); designs = history["designs"].transpose(1, 0); masks = history["masks"]; tau = int((history["n_obs"] ** 2)[0].item())
           inputs = torch.cat([designs, outcomes], dim=-1)  # [T, B, D + Y]
           x = F.relu(self.embedding(inputs))
           x = x.sum(dim=0)  # sum across T
           x = torch.concat([x, masks], dim = -1)
           
        x = self.emitter(x)
        x = self.last_layer[tau](x)
        out = torch.clamp(x, -1, 1)
        return out.unsqueeze(-1)
    

class DADMulti8(nn.Module):
    """
    multiple neural networks for different timesteps
    """
    def __init__(
        self,
        design_size: int,
        y_dim: int,
        context_dim: int,
        embedding_dim: int,
        batch_size: int,
        T: int,
        emitter_dim: int = 4,
    ):
        super().__init__()
        self.design_size = design_size
        # self.embedding = nn.Linear(design_size + y_dim, embedding_dim) # embedding_dim + context_dim
        # self.encoders = nn.ModuleList(
        #     [
        #         nn.Sequential(
        #             nn.Linear(design_size + y_dim, embedding_dim),
        #             nn.ReLU(),
        #             PoolLayer(dim = 0),
        #         )
        #         for _ in range(3)
        #     ]
        # )
        self.emitters = nn.ModuleList(
           [
              nn.Linear(embedding_dim + context_dim, design_size)
              for _ in range(3)
           ]
        )
        # self.emitter = Network([embedding_dim + context_dim, 4])
        self.batch_size = batch_size
        self.context_dim = context_dim
        self.embedding_dim = embedding_dim
    def forward(self, history = None, batch_size: int = None) -> Tensor:

        nn_list = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
        
        if history is None:
           x = torch.empty(1, self.context_dim + self.embedding_dim); tau = 0

        else:
           outcomes = history["outcomes"].transpose(1, 0); designs = history["designs"].transpose(1, 0); masks = history["masks"]; tau = int((history["n_obs"] ** 2)[0].item())
           inputs = torch.cat([designs, outcomes], dim=-1)  # [T, B, D + Y]
           x = F.relu(self.embedding(inputs))
           x = x.sum(dim=0)  # sum across T
           x = torch.concat([x, masks], dim = -1)
           
        x = self.emitters[nn_list[tau]](x)
        out = torch.clamp(x, -1, 1)
        return out.unsqueeze(-1)
    

class DADMulti9(nn.Module):
    """
    Single network for all tau, include tau
    """
    def __init__(
        self,
        design_size: int,
        y_dim: int,
        context_dim: int,
        embedding_dim: int,
        batch_size: int,
        T: int,
        emitter_dim: int = 4,
    ):
        super().__init__()
        self.design_size = design_size
        self.embedding = nn.Linear(design_size + y_dim, embedding_dim) # embedding_dim + context_dim
        self.emitter = Network([embedding_dim + context_dim + 1, 1])
        self.batch_size = batch_size
        self.context_dim = context_dim
        self.embedding_dim = embedding_dim

    def forward(self, history = None, batch_size: int = None) -> Tensor:
        
        if history is None:
           x = torch.empty(1, self.embedding_dim + self.context_dim + 1); tau = 0
        else:
           outcomes = history["outcomes"].transpose(1, 0); designs = history["designs"].transpose(1, 0); masks = history["masks"]; tau = int((history["n_obs"] ** 2)[0].item())
           inputs = torch.cat([designs, outcomes], dim=-1)  # [T, B, D + Y]
           x = F.relu(self.embedding(inputs))
           x = x.sum(dim=0)  # sum across T
           x = torch.concat([x, masks, history["n_obs"]], dim = -1)

        x = self.emitter(x)
        out = torch.clamp(x, -1, 1)
        return out.unsqueeze(-1)
    
class DADMulti10(nn.Module):
    """
    different last layers for all different timesteps THISSSSSS!!! + RELUUUUU
    """
    def __init__(
        self,
        design_size: int,
        y_dim: int,
        context_dim: int,
        embedding_dim: int,
        batch_size: int,
        T: int,
        hidden_dim: int = 4,
    ):
        super().__init__()
        self.design_size = design_size
        self.embedding = nn.Linear(design_size + y_dim, embedding_dim) # embedding_dim + context_dim
        self.emitter = Network([embedding_dim + context_dim, hidden_dim])
        self.last_layer = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, 1)
                )
                for _ in range(T)
            ])
        self.batch_size = batch_size
        self.context_dim = context_dim
        self.embedding_dim = embedding_dim
    def forward(self, history = None, batch_size: int = None) -> Tensor:
        
        if history is None:
           x = torch.empty(1, self.embedding_dim + self.context_dim); tau = 0

        else:
           outcomes = history["outcomes"].transpose(1, 0); designs = history["designs"].transpose(1, 0); masks = history["masks"]; tau = int((history["n_obs"] ** 2)[0].item())
           inputs = torch.cat([designs, outcomes], dim=-1)  # [T, B, D + Y]
           x = F.relu(self.embedding(inputs))
           x = x.sum(dim=0)  # sum across T
           x = torch.concat([x, masks], dim = -1)
           
        x = self.emitter(x)
        x = self.last_layer[tau](x)
        out = torch.clamp(x, -1, 1)
        return out.unsqueeze(-1)
    
class DADMulti11(nn.Module):
    """
    DAD multi 7 for prior samples
    """
    def __init__(
        self,
        design_size: int,
        y_dim: int,
        context_dim: int,
        embedding_dim: int,
        batch_size: int,
        T: int,
        hidden_dim: int = 4,
    ):
        super().__init__()
        self.design_size = design_size
        self.embedding = nn.Linear(design_size + y_dim, embedding_dim) # embedding_dim + context_dim
        self.emitter = nn.Sequential(Network([embedding_dim + context_dim, hidden_dim, hidden_dim, 4]))
        self.batch_size = batch_size
        self.context_dim = context_dim
        self.embedding_dim = embedding_dim
    def forward(self, history = None, batch_size: int = None) -> Tensor:
        
        if history is None:
           x = torch.empty(1, self.embedding_dim + self.context_dim); tau = 0

        else:
           outcomes = history["outcomes"].transpose(1, 0); designs = history["designs"].transpose(1, 0); masks = history["masks"]; tau = int((history["n_obs"] ** 2)[0].item())
           inputs = torch.cat([designs, outcomes], dim=-1)  # [T, B, D + Y]
           x = F.relu(self.embedding(inputs))
           x = x.sum(dim=0)  # sum across T
           x = torch.concat([x, masks], dim = -1)
           
        x = self.emitter(x)
        out = torch.clamp(x, -1, 1)
        return out.unsqueeze(-1)


class DeepAdaptiveDesign(nn.Module):
  def __init__(
      self,
      encoder_net: nn.Module,
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
      outcomes = history["outcomes"].transpose(1, 0); designs = history["designs"].transpose(1, 0) # [B, tau, y_dim] -> [tau, B, y_dim]
      embeddings = torch.stack([self.encoder_net(xi, y) for (xi, y) in zip(designs, outcomes)], dim = 0).sum(dim = 0) # [tau, B, summary_dim] -> [B, summary_dim]
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


class PoolLayer(nn.Module):
    def __init__(self, dim):
        super(PoolLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.sum(x, dim=self.dim)
