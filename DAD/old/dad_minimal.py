import math

import torch
import torch.nn as nn
import torch.distributions as dist

from torch import Tensor
from torch.distributions import Distribution

class SimulatorBasedModel(nn.Module):
  def __init__(self, prior: Distribution, design_net: nn.Module, T: int) -> None:
    super().__init__()
    self.prior = prior
    self.design_net = design_net
    self.T = T

  def outcome_simulator(self, params: Tensor, designs: Tensor) -> Tensor:
    raise NotImplemented

  def forward(self, batch_size: int | None = None) -> tuple[Tensor, Tensor, Tensor]: # call
    prior_params = self.prior.sample(torch.Size([batch_size])) # [B, param_dim]
    designs = []
    outcomes = []
    for t in range(self.T):
      # get design:
      xi = self.design_net(designs, outcomes)
      y = self.outcome_simulator(params=prior_params, designs=xi)

      designs.append(xi)
      outcomes.append(y)

    return prior_params, designs, outcomes

class LikelihoodBasedModel(SimulatorBasedModel):
  def __init__(self, prior: Distribution, design_net: nn.Module, T: int) -> None:
    super().__init__(prior=prior, design_net=design_net, T=T)

  def outcome_likelihood(self, params: Tensor, designs: Tensor) -> Distribution:
    raise NotImplemented

  def outcome_simulator(self, params: Tensor, designs: Tensor) -> Tensor:
    return self.outcome_likelihood(params, designs).sample()
  
  class RandomDesign(nn.Module):
    def __init__(self, design_shape: torch.Size) -> None:
        super().__init__()
        self.design_shape = design_shape
        self.register_buffer("design_mean", torch.zeros(design_shape))
        self.register_buffer("design_sd", torch.ones(design_shape))

    def forward(self, designs=list[Tensor], outcomes=list[Tensor]) -> Tensor:
        if len(outcomes) > 0:
            B = outcomes[0].shape[0]
        else:
            B = 1
            # expad the 0th dim to B:
            design_mean = self.design_mean.expand(B, *self.design_shape)
            design_sd = self.design_sd.expand(B, *self.design_shape)
            
        return dist.Normal(design_mean, design_sd).sample()


class StaticDesign(nn.Module):
  # Start with static design i.e. \pi(data) = \pi() -> policy is independent of
  # the past data
  # Since the API would be the same, it should be easy to extend to DAD, which
  # essentially does two more things: embeds the design-outcome pairs to some
  # fixed-dimensional reprsentation (\xi_t, y_t \mapsto rep_t \in R^emb_dim) and
  # then maps this \rep_t to next design, \xi
  def __init__(self, design_shape: torch.Size, T: int) -> None:
    super().__init__()
    self.design_shape = design_shape
    self.T = T
    self.register_parameter(
        "designs",
        nn.Parameter(torch.randn(self.T, *self.design_shape,  dtype=torch.float32))
    )

  def forward(self, designs=list[Tensor], outcomes=list[Tensor]) -> Tensor:
    t = len(outcomes)
    return self.designs[[t]]

class DeepAdaptiveDesign(nn.Module):
  def __init__(
      self,
      encoder_net: nn.Module,
      decoder_net: nn.Module,
      design_shape: torch.Size
    ) -> None:
    #
    super().__init__()
    self.design_shape = design_shape
    # initialise first design with random normal inititialisation
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
      embeddings = torch.cat([self.encoder_net(xi, y) for (xi, y) in zip(designs, outcomes)]) # TODO aggregate [, dim]
      # get next design
      next_design = self.decoder_net(embeddings)
    return next_design
  
class EncoderNetwork(nn.Module):
    def __init__(
        self,
        design_dim,
        osbervation_dim,
        hidden_dim,
        encoding_dim,
        n_hidden_layers=2,
        activation=nn.Softplus,
    ):
        super().__init__()
        self.encoding_dim = encoding_dim
        self.activation_layer = activation()
        self.input_layer = nn.Linear(design_dim + osbervation_dim, hidden_dim)
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
        inputs = torch.stack([xi, y], dim=-1)
        x = self.input_layer(inputs)
        x = self.activation_layer(x)
        x = self.middle(x)
        x = self.output_layer(x)
        return x

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
    



# Simple linear model
class LinearModel(LikelihoodBasedModel):
  def __init__(
      self,
      prior: Distribution,
      design_net: nn.Module,
      T: int,
      observation_sd: float
      ) -> None:
    super().__init__(prior=prior, design_net=design_net, T=T)
    self.register_buffer("observation_sd", torch.tensor(observation_sd))

  def outcome_likelihood(
      self,
      params: Tensor, # [B, param_dim]
      designs: Tensor # [B, design_dim], here design_dim=param_dim=d
    ) -> Distribution:
    if params.shape!=  designs.shape:
      # print(f"params.shape={params.shape}, designs.shape={designs.shape}")
      assert designs.shape[0] == 1
      designs = designs.expand(params.shape[0], *designs.shape[1:]) # [B, param_dim]

    # multiply [B, param_dim] and [B, param_dim] to get [B, 1] (ie last dim)
    mean_outcome = (designs*params).sum(-1) # [B, 1]
    return dist.Normal(mean_outcome, self.observation_sd)
  

  
dim = 1 # number covariates in the model
T = 5 # number of designs
batch_size = 256

prior = dist.Normal(torch.zeros(dim), torch.ones(dim))
design_net = StaticDesign(design_shape=torch.Size([dim]), T=T)
model = LinearModel(prior, design_net, T=T, observation_sd=1.0)

# params, designs, outcomes = model(batch_size=batch_size)
# print(f"param shape={params.shape}") # [B, param_dim] = [10, 2]
# assert params.shape == torch.Size([batch_size, dim])

# print(f"first design shape={designs[0].shape}")
# assert designs[0].shape == torch.Size([1, dim]) # first design is not batched
# print(f"first outcome shape={outcomes[0].shape}") # outcome dim should be 1
# assert outcomes[0].shape == torch.Size([batch_size])

# print(f"second design shape={designs[0].shape}") # still not batched if Static

# print(f"total designs={len(designs)}, total outcomes={len(outcomes)}")
# assert len(designs) == T and len(outcomes) == T

class MutualInformation(nn.Module):
  def __init__(self, joint_model, batch_size: int) -> None:
    super().__init__()
    self.joint_model = joint_model
    self.batch_size = batch_size

  def forward(self) -> Tensor:
    raise NotImplemented

  def estimate(num_eval_samples) -> float:
    raise NotImplemented

class NestedMonteCarlo(MutualInformation):
  def __init__(
      self,
      joint_model: LikelihoodBasedModel,
      batch_size: int,
      num_negative_samples: int,
      lower_bound: bool = True
      ) -> None:
    super().__init__(joint_model=joint_model, batch_size=batch_size)
    self.num_negative_samples = num_negative_samples #L
    self.lower_bound = lower_bound

  def forward(self) -> Tensor:
    # if lower_bound is false, we compute
    # mean_i { log [p(y_i | primary_i) / sum_j { p(y_i | contrastive_j) }] }
    # otherwise, mean_i { log [ p(y_i | primary_i)  / (p(y_i| primary_i + sum_j { p(y_i | contrastive_j) } ] }

    prior_samples_primary, designs, outcomes = self.joint_model(self.batch_size)
    # we can resuse negative samples
    prior_samples_negative = self.joint_model.prior.sample(
        torch.Size([self.num_negative_samples])
    ).unsqueeze(1) # [num_neg_samples, ...] -> [num_neg_samples, 1, ...]

    # evaluate the logprob of outcomes under the primary:
    logprob_primary = torch.stack([
        self.joint_model.outcome_likelihood(
            prior_samples_primary, xi
        ).log_prob(y) for (xi, y) in zip(designs, outcomes)
    ], dim=0).sum(0) # [T, B] -> [B]

    # evaluate the logprob of outcomes under the contrastive parameter samples:
    logprob_negative = torch.stack([
        self.joint_model.outcome_likelihood(
            prior_samples_negative, xi.unsqueeze(0) # add dim for <num_neg_samples>
        ).log_prob(y.unsqueeze(0)) for (xi, y) in zip(designs, outcomes)
    ], dim=0).sum(0) # [T, num_neg_samples, B] -> [num_neg_samples, B]

    # if lower bound, log_prob primary should be added to the denominator
    if self.lower_bound:
      # concat primary and negative to get [negative_b + 1, B] for the logsumexp
      logprob_negative = torch.cat([
          logprob_negative, logprob_primary.unsqueeze(0)]
      ) # [num_neg_samples + 1, B]
      to_logmeanexp = math.log(self.num_negative_samples + 1)
    else:
      to_logmeanexp = math.log(self.num_negative_samples)

    log_denom = torch.logsumexp(logprob_negative, dim=0) - to_logmeanexp # [B]
    mi = (logprob_primary - log_denom).mean(0) # [B] -> scalar
    return -mi

  def estimate(self) -> float:
    with torch.no_grad():
      loss = self.forward()
    return -loss.item()
  
  # instantiate the MI lower bound objective
pce_loss = NestedMonteCarlo(
    joint_model=model,
    batch_size=batch_size,
    num_negative_samples=512
)

# with the current (randomly initialised) design, we get this MI:
# print(f"MI estimate={pce_loss.estimate()}")

# train designs online (sampling from the model and opimising eg with Adam)
# For the linear model the optimal designs should go to +/- infinity

from torch.optim import Adam
from tqdm import tqdm

# print("\n***")
# print(f"MI estimate before train={pce_loss.estimate()}")
# print(f"Initial designs (before train) {model.design_net.designs}")
num_steps = 10000
optimizer = Adam(model.parameters(), lr=5e-2)
for _ in tqdm(range(num_steps)):
  optimizer.zero_grad()
  loss = pce_loss()
  loss.backward()
  optimizer.step()

# print("\n***")
# print(f"MI estimate after train={pce_loss.estimate()}")
# print(f"Designs after train {model.design_net.designs}")