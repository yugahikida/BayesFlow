import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
import bayesflow as bf
import math
import time

from custom_simulators import LikelihoodBasedModel

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
      joint_model: LikelihoodBasedModel, # joint model with design network
      approximator: bf.approximators,
      batch_size: int,
      num_negative_samples: int,
      lower_bound: bool = True,
      ) -> None:
    super().__init__(joint_model=joint_model, batch_size=batch_size)
    self.num_negative_samples = num_negative_samples # L
    self.lower_bound = lower_bound
    self.approximator = approximator

  def simulate_history(self, n_history, tau) -> dict:
    with torch.no_grad():
      history = self.joint_model.sample(n_history, tau = tau) # simulate h_{\tau} (not backpropagated)
    return history

  def forward(self, history) -> Tensor:
    M = self.joint_model.mask_sampler.possible_masks.shape[0]
    use_pmp = False

    if use_pmp: # Sample from p(m | h_{\tau})
      post_model_prob = self.joint_model.approximate_log_marginal_likelihood(self.batch_size, history, self.approximator) 
    else: # Sample from p(m)
      post_model_prob = torch.full((M,), 1/M)

    prior_samples_primary = []; prior_samples_negative = []; B_list = []; L_list = []; mask_list = []

    for m in range(M):
      masks = self.joint_model.mask_sampler.possible_masks[m]
      B_m = torch.round(post_model_prob[m] * self.batch_size).int() if m != M - 1 else self.batch_size - sum(B_list)
      L_m = torch.round(post_model_prob[m] * self.num_negative_samples).int() if m != M - 1 else self.num_negative_samples - sum(L_list)
      B_list.append(B_m); L_list.append(L_m); mask_list.append(masks.unsqueeze(0).expand(B_m, -1))

      if history is None:
        prior_samples_primary.append(self.joint_model.prior_sampler.sample(masks.unsqueeze(0).expand(B_m, -1))) # [B, B_m]]
        prior_samples_negative.append(self.joint_model.prior_sampler.sample(masks.unsqueeze(0).expand(L_m, -1)))
        n_obs = torch.tensor(self.joint_model.tau_sampler.max_obs, dtype = torch.int32) # T

      else:
        obs_data = {"designs": history["designs"], "outcomes": history["outcomes"], "masks": masks.unsqueeze(0), "n_obs": history["n_obs"]}
        # prior_samples_primary.append(torch.from_numpy(self.approximator.sample(num_samples = B_m, conditions = obs_data)["params"]).squeeze(0))
        # prior_samples_negative.append(torch.from_numpy(self.approximator.sample(num_samples = L_m, conditions = obs_data)["params"]).squeeze(0))
        prior_samples_primary.append(torch.nan_to_num(torch.from_numpy(self.approximator.sample(num_samples = B_m, conditions = obs_data)["params"]).squeeze(0), nan = 0.0))
        prior_samples_negative.append(torch.nan_to_num(torch.from_numpy(self.approximator.sample(num_samples = L_m, conditions = obs_data)["params"]).squeeze(0), nan = 0.0))
        tau = (history["n_obs"] ** 2).int().squeeze(-1)
        n_obs = self.joint_model.tau_sampler.max_obs + 1 - tau # T - \tau
      
    prior_samples_primary = torch.cat(prior_samples_primary, dim=0)
    prior_samples_negative = torch.cat(prior_samples_negative, dim=0).unsqueeze(1)

    # self.joint_model.design_generator.set_freeze(freeze)
    _, _, n_obs, designs, outcomes = self.joint_model(self.batch_size, params = prior_samples_primary, tau = n_obs).values() # simulate h_{(\tau + 1)},..., h_{T}

    logprob_primary = torch.stack(
        [
            self.joint_model.outcome_likelihood(prior_samples_primary, xi.unsqueeze(1)).log_prob(
                y.unsqueeze(1)
            )
            for xi, y in zip(designs.transpose(1, 0), outcomes.transpose(1, 0))
        ],
        dim=0,
    ).sum(0).squeeze()

    logprob_negative = torch.stack(
        [
            self.joint_model.outcome_likelihood(
                prior_samples_negative, xi.unsqueeze(0)
            ).log_prob(y.unsqueeze(0))
            for xi, y in zip(designs.transpose(1, 0), outcomes.transpose(1, 0))
        ],
        dim=0,
    ).sum(0).squeeze()

    if self.lower_bound:
        logprob_negative = torch.cat(
            [logprob_primary.unsqueeze(0), logprob_negative],
            dim=0,
        )
    log_denom = torch.logsumexp(logprob_negative, dim=0) - math.log(
        self.num_negative_samples + self.lower_bound
    )
    mi = (logprob_primary - log_denom).mean(0)

    return -mi

  def estimate(self) -> float:
    with torch.no_grad():
      loss = self.forward(history = None)
    return -loss.item()