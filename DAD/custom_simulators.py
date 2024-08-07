import bayesflow as bf
from bayesflow.simulators.simulator import Simulator
import torch
from torch import Tensor
from torch.distributions import Distribution
import torch.nn as nn
from typing import Callable
from design_networks import DeepAdaptiveDesign

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

        B = params.shape[0]
        
        designs = []
        outcomes = []

        for t in range(tau):
            if t == 0 and isinstance(self.design_generator, DeepAdaptiveDesign): 
                xi = self.design_generator(history = None, batch_size = B).view(1, 1, 1).repeat(B, 1, 1) # [B, 1, xi_dim] # expand initial design

            else:
                designs_t = torch.stack(designs, dim=1).squeeze(-1) if len(designs) != 0 else 0 # [B, tau, design_dim] or (0 in case of using random design)
                outcomes_t = torch.stack(outcomes, dim=1).squeeze(-1) if len(designs) != 0 else 0 # [B, tau, design_dim] or 0
                xi = self.design_generator(history = {"designs": designs_t, "outcomes": outcomes_t}, batch_size = B)  # [B, 1, xi_dim] 

            y = self.outcome_simulator(params=params, xi=xi) # [B, tau, y_dim]

            # designs = torch.cat((designs, xi), dim=1)
            # outcomes = torch.cat((outcomes, y), dim=1)
            designs.append(xi)
            outcomes.append(y)

        designs = torch.stack(designs, dim=1).squeeze(-1) # [B, tau, design_dim]
        outcomes = torch.stack(outcomes, dim=1).squeeze(-1) # [B, tau, design_dim]

        n_obs = torch.sqrt(tau).repeat(batch_size[0]).unsqueeze(1) # [B, 1]

        out = {"masks": masks, "params": params, "n_obs": n_obs, "designs": designs, "outcomes": outcomes} # ]

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
    
    def approximate_log_marginal_likelihood(self, B: int, history: dict, approximator: bf.Approximator) -> Tensor:

        possible_masks = self.mask_sampler.possible_masks
        M = possible_masks.shape[0]

        log_marginal_likelihood = []
        
        for m in range(M):
            masks = possible_masks[m]
            obs_data = {"designs": history["designs"], "outcomes": history["outcomes"], "masks": masks.unsqueeze(0), "n_obs": history["n_obs"]}
            post_samples = approximator.sample((1, B), obs_data)["params"].to('cpu')

            first_term = torch.stack([self.outcome_likelihood(theta.unsqueeze(0), history["designs"], self.simulator_var).log_prob(history["outcomes"]) for theta in post_samples]) # [B]  p(y | theta, xi) for B posterior samples
            second_term = self.prior_sampler.log_prob(post_samples, obs_data["masks"]) # [B]

            obs_data_rep = {"params": post_samples, "designs": history["designs"].repeat(B, 1, 1), "outcomes": history["outcomes"].repeat(B, 1, 1), "masks": masks.unsqueeze(0).repeat(B, 1), "n_obs": history["n_obs"].repeat(B, 1)}
            third_term = approximator.log_prob(obs_data_rep) # [B] 

            log_marginal_likelihood_m = (first_term + second_term - third_term).mean() # [B] -> [1]
            log_marginal_likelihood.append(log_marginal_likelihood_m) 

        log_marginal_likelihood = torch.stack(log_marginal_likelihood, dim = 0) # list -> [M]

        return log_marginal_likelihood
    
    def posterior_model_prob(self, B: int, history: dict, approximator: bf.Approximator) -> Tensor:
        log_marginal_likelihood = self.approximate_log_marginal_likelihood(B, history, approximator) # [M]

        logm_max = torch.max(log_marginal_likelihood)
        logm_sum = torch.exp(log_marginal_likelihood - logm_max).sum()
        
        log_normalizer =  logm_max + torch.log(logm_sum)
        log_pmp = log_marginal_likelihood - log_normalizer

        return torch.exp(log_pmp)
    

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
        return torch.randint(self.min_obs, self.max_obs, (1,))