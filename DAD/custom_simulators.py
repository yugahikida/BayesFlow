import bayesflow as bf
from bayesflow.simulators.simulator import Simulator
import torch
from torch import Tensor
from torch.distributions import Distribution
import torch.nn as nn
from typing import Callable
from design_networks import DeepAdaptiveDesign
import math

class GenericSimulator(Simulator, nn.Module):
    def __init__(self, mask_sampler: Callable, prior_sampler: Callable, tau_sampler: Callable, design_generator: nn.Module, sim_vars: dict):
        super().__init__()
        self.mask_sampler = mask_sampler
        self.prior_sampler = prior_sampler
        self.tau_sampler = tau_sampler
        self.design_generator = design_generator
        self.sim_vars = sim_vars

    def forward(self, batch_size: int, masks: Tensor = None, params : Tensor = None, tau : int = None, **kwargs) -> dict[str, Tensor]:
        return self.sample(batch_size, masks, params, tau, **kwargs)

    def sample(self, batch_size: int, masks: Tensor = None, params : Tensor = None, tau : int = None, **kwargs) -> dict[str, Tensor]:            
            
        masks = masks.expand(batch_size, -1) if masks is not None else self.mask_sampler(batch_size) # [B, mask_dim]
        params = params if params is not None else self.prior_sampler.sample(masks)
        tau = torch.tensor([tau]) if tau is not None else self.tau_sampler()
        if tau == 0: return None
        designs = []; outcomes = []

        for t in range(tau):
            if t == 0:
                xi = self.design_generator(history = None, batch_size = batch_size).expand(batch_size, 1, 1) # [B, 1, xi_dim] # expand initial design
            else:
                designs_t = torch.stack(designs, dim=1).squeeze(-1) if len(designs) != 0 else 0 # [B, tau, design_dim] or (None in case of using random design)
                outcomes_t = torch.stack(outcomes, dim=1).squeeze(-1) if len(designs) != 0 else 0 # [B, tau, design_dim] or None
                xi = self.design_generator(history = {"masks": masks, "n_obs": torch.tensor([math.sqrt(t)]).expand(batch_size, 1), "designs": designs_t, "outcomes": outcomes_t}, batch_size = batch_size)  # [B, 1, xi_dim] 
            y = self.outcome_simulator(params=params, xi=xi) # [B, tau, y_dim]
            designs.append(xi); outcomes.append(y)

        designs = torch.stack(designs, dim=1).squeeze(-1) # [B, tau, design_dim]
        outcomes = torch.stack(outcomes, dim=1).squeeze(-1) # [B, tau, design_dim]
        n_obs = torch.sqrt(tau).repeat(batch_size).unsqueeze(1) # [B, 1]

        return  {"masks": masks, "params": params, "n_obs": n_obs, "designs": designs, "outcomes": outcomes}
    
    def outcome_simulator(self, params: Tensor, xi: Tensor) -> Tensor:
        raise NotImplementedError
    
class LikelihoodBasedModel(GenericSimulator):
    def __init__(self, mask_sampler, prior_sampler, tau_sampler, design_generator, sim_vars) -> None:
        super().__init__(mask_sampler, prior_sampler, tau_sampler, design_generator, sim_vars)

    def outcome_likelihood(self, params: Tensor, xi: Tensor) -> Distribution:
        raise NotImplementedError
    
    def outcome_simulator(self, params: Tensor, xi: Tensor) -> Tensor:
        return self.outcome_likelihood(params, xi).rsample()
    
    # def approximate_log_marginal_likelihood(self, batch_size: int, history: dict, approximator: bf.approximators) -> Tensor:
    #     possible_masks = self.mask_sampler.possible_masks
    #     M = possible_masks.shape[0]
    #     log_marginal_likelihood = []
        
    #     for m in range(M):
    #         masks = possible_masks[m]
    #         obs_data = {"designs": history["designs"], "outcomes": history["outcomes"], "masks": masks.unsqueeze(0), "n_obs": history["n_obs"]}
    #         post_samples = approximator.sample((1, batch_size), obs_data)["params"].to('cpu')

    #         first_term = torch.stack([self.outcome_likelihood(theta.unsqueeze(0), history["designs"], self.sim_vars).log_prob(history["outcomes"]) for theta in post_samples]) # [B]  p(y | theta, xi) for B posterior samples
    #         second_term = self.prior_sampler.log_prob(post_samples, obs_data["masks"]) # [B]

    #         obs_data_rep = {"params": post_samples, "designs": history["designs"].repeat(batch_size, 1, 1), "outcomes": history["outcomes"].repeat(batch_size, 1, 1), "masks": masks.unsqueeze(0).repeat(B, 1), "n_obs": history["n_obs"].repeat(B, 1)}
    #         third_term = approximator.log_prob(obs_data_rep) # [B] 

    #         log_marginal_likelihood_m = (first_term + second_term - third_term).mean() # [B] -> [1]
    #         log_marginal_likelihood.append(log_marginal_likelihood_m)

    #     log_marginal_likelihood = torch.stack(log_marginal_likelihood, dim = 0) # list -> [M]

    #     return log_marginal_likelihood
    
    # def posterior_model_prob(self, batch_size: int, history: dict, approximator: bf.approximators) -> Tensor:
    #     log_marginal_likelihood = self.approximate_log_marginal_likelihood(batch_size, history, approximator) # [M]

    #     logm_max = torch.max(log_marginal_likelihood)
    #     logm_sum = torch.exp(log_marginal_likelihood - logm_max).sum()
        
    #     log_normalizer =  logm_max + torch.log(logm_sum)
    #     log_pmp = log_marginal_likelihood - log_normalizer

    #     return torch.exp(log_pmp)
    

class ParameterMask:
    def __init__(self, num_parameters: int = 4, possible_masks: Tensor = None) -> None:
        default_mask = torch.tril(torch.ones((num_parameters, num_parameters)))
        self.num_parameters = num_parameters
        self.possible_masks = torch.tensor(possible_masks, dtype=torch.float32) if possible_masks is not None else default_mask

    def __call__(self, batch_size: int) -> Tensor:
        index_samples = torch.randint(0, self.possible_masks.shape[0], torch.Size([batch_size]), dtype=torch.long)
        out_mask = self.possible_masks[index_samples]
        return out_mask
    
class Prior():
    def __init__(self) -> None:
        super().__init__()
    
    def dist(self, masks: Tensor) -> Distribution:
        raise NotImplementedError

    def sample(self, masks: Tensor) -> Tensor:
        return self.dist(masks).sample()

    def log_prob(self, params: Tensor, masks: Tensor) -> Tensor:
        return self.dist(masks).log_prob(params)
    
class RandomNumObs():
    def __init__(self, min_obs : int, max_obs: int) -> Tensor:
        self.min_obs = min_obs
        self.max_obs = max_obs # T

    def __call__(self):
        return torch.randint(self.min_obs, self.max_obs + 1, (1,))