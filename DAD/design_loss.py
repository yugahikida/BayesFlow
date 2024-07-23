import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
import bayesflow as bf

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
      joint_model: LikelihoodBasedModel,
      amortized_posterior: bf.networks,
      batch_size: torch.Size,
      num_negative_samples: int,
      lower_bound: bool = True
      ) -> None:
    super().__init__(joint_model=joint_model, batch_size=batch_size)
    self.num_negative_samples = num_negative_samples # L
    self.lower_bound = lower_bound
    self.amortized_posterior = amortized_posterior

  def forward(self) -> Tensor:

    # simulate one trajectory of history.
    # masks_h, params_h, tau, xi_h, y_h = self.joint_model.sample(torch.Size([1])).values() # simulate h_{\tau}
    history = self.joint_model.sample(torch.Size([1])) # simulate h_{\tau}

    post_model_prob, post_samples_list = self.joint_model.approximate_log_marginal_likelihood(history) # p(m | h_{\tau})

    # masks = torch.from_numpy(np.random.choice(self.joint_model.context_sampler.possible_masks, 
    #                            size = self.batch_size, p = post_model_prob)) # m ~ p(m | h_{\tau})

    M = self.joint_model.context_sampler.possible_masks.shape[0]
    
    for m in range(M):
      B_m = post_model_prob[m] * self.batch_size # p(m | h_{\tau}) * B
      # obs_data = {"designs": history["designs"], "outcomes": history["outcomes"], "masks": history["masks"], "n_obs": history["n_obs"]}
      # prior_samples_primary = self.amortized_posterior.sample((1, B_m), obs_data) # p(\theta_m | m, h_{\tau})
      prior_samples_primary = post_samples_list[m][torch.randperm(self.batch_size)[:B_m]]

    n_obs = self.joint_model.tau_sampler.max_obs - history["n_obs"]

    _, _, _, designs, outcomes = self.joint_model.sample((1, self.batch_size), params = prior_samples_primary, tau = n_obs).values() # simulate h_{(\tau + 1)},..., h_{T}

    # we can resuse negative samples
    prior_samples_negative = self.amortized_posterior.sample(
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

    print("nagative param", prior_samples_negative.shape)
    print("one design", designs[0].unsqueeze(0).shape)
    print("one outcomes", outcomes[0].unsqueeze(0).shape)

    # if lower bound, log_prob primary should be added to the denominator
    if self.lower_bound:
      # concat primary and negative to get [negative_b + 1, B] for the logsumexp
      logprob_negative = torch.cat([
          logprob_negative, logprob_primary.unsqueeze(0)]
      ) # [num_neg_samples + 1, B]
      to_logmeanexp = torch.log(self.num_negative_samples + 1)
    else:
      to_logmeanexp = torch.log(self.num_negative_samples)

    log_denom = torch.logsumexp(logprob_negative, dim=0) - to_logmeanexp # [B]
    mi = (logprob_primary - log_denom).mean(0) # [B] -> scalar
    return -mi

  def estimate(self) -> float:
    with torch.no_grad():
      loss = self.forward()
    return -loss.item()