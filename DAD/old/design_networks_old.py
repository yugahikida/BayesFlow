import torch.nn as nn

# old version of emitter network to load weights trained with this

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
  
class ScalingLayer(nn.Module):
    def __init__(self, scale_factor):
        super(ScalingLayer, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return x * self.scale_factor
    