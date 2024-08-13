import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
import bayesflow as bf
import math
import time

from custom_simulators import LikelihoodBasedModel

class MutualInformation(nn.Module):
  def __init__(self, joint_model, batch_shape: torch.Size) -> None:
    super().__init__()
    self.joint_model = joint_model
    self.batch_shape = batch_shape

  def forward(self) -> Tensor:
    raise NotImplemented

  def estimate(num_eval_samples) -> float:
    raise NotImplemented

class NestedMonteCarlo(MutualInformation):
  def __init__(
      self,
      joint_model: LikelihoodBasedModel, # joint model with design network
      approximator: bf.Approximator,
      batch_shape: torch.Size,
      num_negative_samples: int,
      lower_bound: bool = True
      ) -> None:
    super().__init__(joint_model=joint_model, batch_shape=batch_shape)
    self.num_negative_samples = num_negative_samples # L
    self.lower_bound = lower_bound
    self.approximator = approximator

  def forward(self) -> Tensor:

    # simulate one trajectory of history.
    history = self.joint_model.sample(torch.Size([1])) # simulate h_{\tau}  dataset

    M = self.joint_model.mask_sampler.possible_masks.shape[0]

    use_pmp = False

    if use_pmp: # Sample from p(m | h_{\tau})
      post_model_prob = self.joint_model.approximate_log_marginal_likelihood(self.batch_shape[0], history, self.approximator) 

    else:
      post_model_prob = torch.full((M,), 1/M)

    B = self.batch_shape[0]
    param_dim = history["params"].shape[-1]

    prior_samples_primary = []; prior_samples_negative = []

    B_list = []; L_list = []

    for m in range(M):
      masks = self.joint_model.mask_sampler.possible_masks[m]

      B_m = torch.round(post_model_prob[m] * self.batch_shape[0]).int() if m != M else self.batch_shape[0] - torch.sum(B_list)
      L_m = torch.round(post_model_prob[m] * self.num_negative_samples).int() if m != M else self.num_negative_samples - torch.sum(L_list)
      obs_data = {"designs": history["designs"], "outcomes": history["outcomes"], "masks": masks.unsqueeze(0), "n_obs": history["n_obs"]}

      B_list.append(B_m); L_list.append(L_m)

      prior_samples_primary.append(self.approximator.sample((1, B_m), obs_data)["params"].to('cpu')) 
      prior_samples_negative.append(self.approximator.sample((1, L_m), obs_data)["params"].to('cpu'))

    n_obs = self.joint_model.tau_sampler.max_obs - (history["n_obs"] ** 2).int().squeeze(-1) # T - \tau
    prior_samples_primary = torch.cat(prior_samples_primary, dim=0)
    prior_samples_negative = torch.cat(prior_samples_negative, dim=0)

    _, _, _, designs, outcomes = self.joint_model(self.batch_shape, params = prior_samples_primary, tau = n_obs).values() # simulate h_{(\tau + 1)},..., h_{T}
    
    # params: [B, param_dim], xi: [B, 1, xi_dim]

    logprob_primary = torch.stack([
        self.joint_model.outcome_likelihood(
            prior_samples_primary, xi.unsqueeze(1), self.joint_model.simulator_var # add dimensiont for n_obs
        ).log_prob(y.unsqueeze(1)) for (xi, y) in zip(designs.transpose(1, 0), outcomes.transpose(1, 0))
    ], dim=0).sum(0).squeeze() # should work unless batch size is 1
  

    logprob_negative = torch.stack([
        self.joint_model.outcome_likelihood(
            prior_samples_negative.unsqueeze(0), xi.unsqueeze(1).unsqueeze(0), self.joint_model.simulator_var # add dim for n_obs and num_neg_samples
        ).log_prob(y.unsqueeze(1).unsqueeze(0)) for (xi, y) in zip(designs.transpose(1, 0), outcomes.transpose(1, 0))
    ], dim=0).squeeze((1, -1)).sum(0) # [T, 1, num_neg_samples, B, 1] -> [T, num_neg_samples, B] -> [num_neg_samples, B]
    
    # if lower bound, log_prob primary should be added to the denominator
    if self.lower_bound:
      # concat primary and negative to get [num_neg_samples + 1, B] for the logsumexp
      logprob_negative = torch.cat([
          logprob_negative, logprob_primary.unsqueeze(0)
      ]) # [num_neg_samples + 1, B]
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