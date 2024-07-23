import bayesflow as bf
from bayesflow.simulators.simulator import Simulator
import torch
from torch import Tensor
from torch.distributions import Distribution
import torch.nn as nn
from typing import Callable

class MyGenericSimulator(Simulator):
    def __init__(self, mask_sampler: Callable, prior_sampler: Callable, tau_sampler: Callable, design_generator: nn.Module, simulator_var: dict):
        self.mask_sampler = mask_sampler
        self.prior_sampler = prior_sampler
        self.tau_sampler = tau_sampler
        self.design_generator = design_generator
        self.simulator_var = simulator_var

    def sample(self, batch_size: torch.Size, masks: Tensor = None, params : Tensor = None, tau : int = None, **kwargs) -> dict[str, Tensor]:            
            
        masks = None if params is not None else self.mask_sampler(batch_size)
        params = params if params is not None else self.prior_sampler.sample(masks)
        tau = tau if tau is not None else self.tau_sampler()
        
        designs = []
        outcomes = []

        for t in range(tau):
            xi = self.design_generator(batch_size)

            # if params.shape[0] != xi.shape[0]: # for initial design
            #     xi = xi.repeat(params.shape[0], 1)

            y = self.outcome_simulator(params=params, xi=xi)

            designs.append(xi)
            outcomes.append(y)

        designs = torch.stack(designs, dim=1).unsqueeze(-1)  #  [B, tau, 1]
        outcomes = torch.stack(outcomes, dim=1) # [B, tau, 1]
        n_obs = torch.sqrt(tau).repeat(batch_size).unsqueeze(1) # [B, 1]

        out = {"masks": masks, "params": params, "n_obs": n_obs, "designs": designs, "outcomes": outcomes}

        return out
    
    def outcome_simulator(self, params: Tensor, xi: Tensor) -> Tensor:
        raise NotImplementedError
    
class LikelihoodBasedModel(MyGenericSimulator):
    def __init__(self, mask_sampler, prior_sampler, tau_sampler, design_generator, simulator_var) -> None:
        super().__init__(mask_sampler, prior_sampler, tau_sampler, design_generator, simulator_var)

    def outcome_likelihood(self, params: Tensor, xi: Tensor) -> Distribution:
        raise NotImplementedError
    
    def outcome_simulator(self, params: Tensor, xi: Tensor) -> Tensor:
        return self.outcome_likelihood(params, xi, self.simulator_var).sample()
    
    def approximate_log_marginal_likelihood(self, B: int, history: dict, approx_posterior: bf.networks) -> Tensor:

        possible_masks = self.mask_sampler.possible_masks
        M = possible_masks.shape[0]

        marginal_likelihood = []
        post_samples_list = []
        
        for m in range(M):
            masks = possible_masks[m]
            obs_data = {"designs": history["designs"], "outcomes": history["outcomes"], "masks": masks.unsqueeze(0), "n_obs": history["n_obs"]}
            post_samples = approx_posterior.sample((1, B), obs_data)["params"].to('cpu')

            first_term = torch.stack([self.outcome_likelihood(theta, xi, self.simulator_var).log_prob(y.unsqueeze(-1)).sum() for theta, xi, y in zip(post_samples, obs_data["designs"], obs_data["outcomes"])]).sum() # [B, tau] -> [B] -> [1]
            second_term = self.prior_sampler.log_prob(post_samples, masks.unsqueeze(0)).sum() # [B]  -> [1]

            obs_data_rep = {"params": post_samples, "designs": history["designs"].repeat(B, 1, 1), "outcomes": history["outcomes"].repeat(B, 1, 1), "masks": masks.unsqueeze(0).repeat(B, 1), "n_obs": history["n_obs"].repeat(B, 1)}
            third_term = approx_posterior.log_prob(obs_data_rep).sum() # [B] -> [1]

            marginal_likelihood_m = torch.exp(first_term + second_term - third_term)
            marginal_likelihood.append(marginal_likelihood_m)

            post_samples_list.append(post_samples)

        marginal_likelihood = torch.stack(marginal_likelihood, dim = 0) / torch.stack(marginal_likelihood, dim = 0).sum()
        
        return marginal_likelihood, post_samples_list
    

class ParameterMask:
    def __init__(self, num_parameters: int = 4, possible_masks: Tensor = None) -> None:
        default_mask = torch.tril(torch.ones((num_parameters, num_parameters)))
        self.num_parameters = num_parameters
        self.possible_masks = torch.tensor(possible_masks, dtype=torch.float32) if possible_masks is not None else default_mask

    def __call__(self, batch_shape: torch.Size) -> Tensor:
        index_samples = torch.randint(0, self.possible_masks.shape[0], batch_shape, dtype=torch.long)
        out_mask = self.possible_masks[index_samples]

        return out_mask
    
class Prior():
    def __init__(self) -> None:
        super().__init__()
    
    def dist(self, masks: Tensor) -> [Distribution]:
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