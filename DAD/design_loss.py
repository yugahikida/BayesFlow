import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
import bayesflow as bf
import math

from custom_simulators import LikelihoodBasedModel

class MutualInformation(nn.Module):
  def __init__(self, joint_model, batch_size: torch.Size) -> None:
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
      approximator: bf.Approximator,
      batch_size: int,
      num_negative_samples: int,
      lower_bound: bool = True
      ) -> None:
    super().__init__(joint_model=joint_model, batch_size=batch_size)
    self.num_negative_samples = num_negative_samples # L
    self.lower_bound = lower_bound
    self.approximator = approximator

  def forward(self) -> Tensor:

    # simulate one trajectory of history.
    history = self.joint_model.sample(torch.Size([1])) # simulate h_{\tau}

    post_model_prob, post_samples_list = self.joint_model.approximate_log_marginal_likelihood(self.batch_size[0], history, self.approximator) # p(m | h_{\tau})

    M = self.joint_model.mask_sampler.possible_masks.shape[0]
    
    B = self.batch_size[0]
    param_dim = history["params"].shape[-1]

    prior_model_primary = torch.empty((0, M))
    prior_model_negative = torch.empty((0, M))

    prior_samples_primary = torch.empty((0, param_dim)) 
    prior_samples_negative = torch.empty((0, param_dim))
    
    for m in range(M):
      masks = self.joint_model.mask_sampler.possible_masks[m]
      B_m = torch.round(post_model_prob[m] * self.batch_size[0]).int() # p(m | h_{\tau}) * B
      L_m = torch.round(post_model_prob[m] * self.num_negative_samples).int()
      obs_data = {"designs": history["designs"], "outcomes": history["outcomes"], "masks": masks.unsqueeze(0), "n_obs": history["n_obs"]}

      prior_model_primary_m = masks.unsqueeze(0).repeat(B_m, 1)
      prior_model_primary = torch.cat((prior_model_primary, prior_model_primary_m), dim=0)

      prior_samples_primary_m = post_samples_list[m][torch.randperm(B)[:B_m]] # reuse posterior samples used to obtain posterior model probabilities.
      prior_samples_primary = torch.cat((prior_samples_primary, prior_samples_primary_m), dim=0)

      prior_model_negative_m = masks.unsqueeze(0).repeat(L_m, 1)
      prior_model_negative = torch.cat((prior_model_negative, prior_model_negative_m), dim=0)

      prior_samples_negative_m = self.approximator.sample((1, L_m), obs_data)["params"].to('cpu')
      prior_samples_negative = torch.cat((prior_samples_negative, prior_samples_negative_m), dim=0)

    n_obs = self.joint_model.tau_sampler.max_obs - (history["n_obs"] ** 2).int().squeeze(-1) # T - \tau

    _, _, _, designs, outcomes = self.joint_model.sample(self.batch_size, params = prior_samples_primary, tau = n_obs).values() # simulate h_{(\tau + 1)},..., h_{T}


    # evaluate the logprob of outcomes under the primary:

    # obs_data_rep_primary = {"params": prior_samples_primary, "designs": designs, "outcomes": outcomes, "masks": prior_model_primary, "n_obs": history["n_obs"].repeat(B, 1)}
    # logprob_primary = self.approximator.log_prob(obs_data_rep_primary).sum(0) # [B] -> [1]

    logprob_primary = torch.stack([
      self.joint_model.outcome_likelihood(
      theta.unsqueeze(0), xi.unsqueeze(0), self.joint_model.simulator_var
      ).log_prob(y.unsqueeze(0))
      for (theta, xi, y) in zip(prior_samples_primary, designs, outcomes)
      ], dim = 0).squeeze(1).squeeze(-1).sum(1) # [B, T - tau] -> [B]


    # evaluate the logprob of outcomes under the contrastive parameter samples:
    # obs_data_rep_negative = {"params": prior_samples_negative, "designs": history["designs"].repeat(self.num_negative_samples, 1, 1), "outcomes": history["outcomes"].repeat(self.num_negative_samples, 1, 1), "masks": prior_model_negative, "n_obs": history["n_obs"].repeat(self.num_negative_samples, 1)}
    # logprob_negative = self.approximator.log_prob(obs_data_rep_negative).sum(0, keepdim = True) # [B] -> [1]
      

    tmp_s = list([self.joint_model.outcome_likelihood(
              theta.unsqueeze(0), xi.unsqueeze(0), self.joint_model.simulator_var
              ).log_prob(y.unsqueeze(0)) for (theta, xi, y) in zip(*(
                theta_s, xi_s, y_s))] for (theta_s, xi_s, y_s) in zip(prior_samples_negative.unsqueeze(0).repeat(B, 1, 1), 
                                                                designs.unsqueeze(1).repeat(1, self.num_negative_samples, 1, 1), 
                                                                outcomes.unsqueeze(1).repeat(1, self.num_negative_samples, 1, 1)))
    
    logprob_negative = torch.stack([torch.stack(tmp) for tmp in tmp_s], dim=0).squeeze().sum(-1).t() # [B, L, n_obs] -> [B, L] -> [L, B]
    

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