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
      scale_list: list = None,
      ) -> None:
    super().__init__(joint_model=joint_model, batch_size=batch_size)
    self.num_negative_samples = num_negative_samples # L
    self.lower_bound = lower_bound
    self.approximator = approximator
    self.scale_list = scale_list

  def simulate_history(self, n_history, tau) -> dict:
    with torch.no_grad():
      history = self.joint_model.sample(n_history, tau = tau) # simulate h_{\tau} (not backpropagated)
    return history

  def forward(self, history) -> Tensor:
    # history = self.simulate_history(n_history, tau)

    # simulate one trajectory of history.
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
        prior_samples_primary.append(self.approximator.sample((1, B_m), obs_data)["params"].to('cpu')) 
        prior_samples_negative.append(self.approximator.sample((1, L_m), obs_data)["params"].to('cpu'))
        tau = (history["n_obs"] ** 2).int().squeeze(-1)
        n_obs = self.joint_model.tau_sampler.max_obs - tau # T - \tau
      
    prior_samples_primary = torch.cat(prior_samples_primary, dim=0)
    prior_samples_negative = torch.cat(prior_samples_negative, dim=0).unsqueeze(1)
    # mask_primary = torch.cat(mask_list, dim = 0)

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

    if self.scale_list is not None:
      mi = self.scale_lists[tau] + mi

    # if eval_mode:
    #   with torch.no_grad():
    #     eval_dict = {"mi": -mi.item(), "designs": designs, "outcomes": outcomes, "params": prior_samples_primary, "masks": mask_primary, "n_obs": n_obs}
    #   return eval_dict

    return -mi

    # T = self.joint_model.tau_sampler.max_obs

    # if history is not None:
    #   # weights = np.arange(T) + 1
    #   # normalised_weights = weights / np.sum(weights)
    #   return -mi # * normalised_weights[tau]
    
    # else:
    #   return -mi # / np.sum(np.arange(T) + 1)

  def estimate(self) -> float:
    with torch.no_grad():
      loss = self.forward(history = None)
    return -loss.item()
  
  # def evaluate(self) -> dict:
  #   with torch.no_grad():
  #     eval_dict = self.forward(history = None, eval_mode = True)
  #   return eval_dict



    


# class NestedMonteCarloMultipleHistory(MutualInformation):
#   def __init__(
#       self,
#       joint_model: LikelihoodBasedModel, # joint model with design network
#       approximator: bf.approximators,
#       batch_shape: torch.Size,
#       num_negative_samples: int,
#       lower_bound: bool = True,
#       ) -> None:
#     super().__init__(joint_model=joint_model, batch_shape=batch_shape)
#     self.num_negative_samples = num_negative_samples # L
#     self.lower_bound = lower_bound
#     self.approximator = approximator

#   def simulate_history(self, n_history = 1, tau = None) -> dict:
#     # simulate trajectories of history.
#     history = self.joint_model.sample(torch.Size([n_history]), tau = tau) # simulate h_{\tau}  dataset
#     return history

#   def forward(self, history) -> Tensor:
#     M = self.joint_model.mask_sampler.possible_masks.shape[0]
#     use_pmp = False

#     if history is not None:
#       n_history = history["designs"].shape[0] # number of history trajectories to simululate for one renewal of nn weights
    
#     else:
#       n_history = 1

#     param_dim = 4 # TODO: change this to be more general

#     if use_pmp: # Sample from p(m | h_{\tau})
#       post_model_prob = self.joint_model.approximate_log_marginal_likelihood(self.batch_shape[0], history, self.approximator) 

#     else:
#       post_model_prob = torch.full((M,), 1/M)

#     prior_samples_primary = []; prior_samples_negative = []

#     B_list = []; L_list = []

#     for m in range(M):
#       masks = self.joint_model.mask_sampler.possible_masks[m]

#       B_m = torch.round(post_model_prob[m] * self.batch_shape[0]).int() if m != M else self.batch_shape[0] - torch.sum(B_list)
#       L_m = torch.round(post_model_prob[m] * self.num_negative_samples).int() if m != M else self.num_negative_samples - torch.sum(L_list)
#       B_list.append(B_m); L_list.append(L_m)

#       if history is None:
#         tau = 0
#         prior_samples_primary.append(self.joint_model.prior_sampler.sample(masks.unsqueeze(0).expand(B_m, -1))) # [B, B_m]]
#         prior_samples_negative.append(self.joint_model.prior_sampler.sample(masks.unsqueeze(0).expand(L_m, -1)))
#         n_obs = torch.tensor(self.joint_model.tau_sampler.max_obs, dtype = torch.int32) # T

#       else:
#         tau = (history["n_obs"]**2).int().squeeze(-1)
#         obs_data = {"designs": history["designs"], "outcomes": history["outcomes"], "masks": masks.unsqueeze(0).repeat(n_history, 1), "n_obs": history["n_obs"]}
#         prior_samples_primary.append(self.approximator.sample((n_history, B_m), data = obs_data)["params"].to('cpu'))
#         prior_samples_negative.append(self.approximator.sample((n_history, L_m), data = obs_data)["params"].to('cpu')) 
#         n_obs = self.joint_model.tau_sampler.max_obs - tau # T - \tau

#     prior_samples_primary = torch.stack(prior_samples_primary, dim=1).view(n_history, self.batch_shape[0], param_dim) # [n_history, B, param_dim]
#     prior_samples_negative = torch.stack(prior_samples_negative, dim=1).view(n_history, self.num_negative_samples, param_dim)

#     designs = []; outcomes = []

#     for i in range(n_history):
#       _, _, _, d, o = self.joint_model(self.batch_shape, params = prior_samples_primary[i], tau = n_obs).values() # simulate h_{(\tau + 1)},..., h_{T}
#       designs.append(d); outcomes.append(o)
    
#     # params: [B, param_dim], xi: [B, 1, xi_dim]
#     designs = torch.stack(designs, dim=0)
#     outcomes = torch.stack(outcomes, dim=0)

#     logprob_primary = torch.stack([torch.stack([
#         self.joint_model.outcome_likelihood(
#             primary, xi.unsqueeze(1), self.joint_model.sim_vars # add dimensiont for n_obs
#         ).log_prob(y.unsqueeze(1)) for (primary, xi, y) in zip(prior_samples_primary, designs_i, outcomes_i)]) 
#                                    for (designs_i, outcomes_i) in zip(designs.permute(2, 0, 1, 3), outcomes.permute(2, 0, 1, 3))
#     ], dim=0).sum(0).squeeze(-1, -2).t() # should work unless batch size is 1

#     logprob_negative = torch.stack([torch.stack([
#         self.joint_model.outcome_likelihood(
#             primary.unsqueeze(0), xi.unsqueeze(1).unsqueeze(0), self.joint_model.sim_vars # add dim for n_obs and num_neg_samples
#         ).log_prob(y.unsqueeze(1).unsqueeze(0)) for (primary, xi, y) in zip(prior_samples_primary, designs_i, outcomes_i)])
#                                                 for (designs_i, outcomes_i) in zip(designs.permute(2, 0, 1, 3), outcomes.permute(2, 0, 1, 3))
#                                                 ], dim=0).squeeze((2, -1)).sum(0).permute(1, 2, 0)
    
#     # if lower bound, log_prob primary should be added to the denominator
#     if self.lower_bound:
#       # concat primary and negative to get [num_neg_samples + 1, B] for the logsumexp
#       logprob_negative = torch.cat([
#           logprob_negative, logprob_primary.unsqueeze(0) # TODO: add MI up to time tau / expontinaly weight
#       ])

#       to_logmeanexp = math.log(self.num_negative_samples + 1)
#     else:
#       to_logmeanexp = math.log(self.num_negative_samples)

#     log_denom = torch.logsumexp(logprob_negative, dim=0) - to_logmeanexp # [B]
#     mi = (logprob_primary - log_denom).mean(0).mean(0) # [B] -> scalar

#     T = self.joint_model.tau_sampler.max_obs
#     weights = np.arange(T) + 1
#     normalised_weights = weights / np.sum(weights)

#     #weights = np.exp(np.arange(T))
#     #normalised_weights = weights / np.sum(weights)

#     return -mi * normalised_weights[tau]

#   def estimate(self) -> float:
#     with torch.no_grad():
#       loss = self.forward()
#     return -loss.item()
