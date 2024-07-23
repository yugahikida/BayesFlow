import os
os.environ["KERAS_BACKEND"] = "torch"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import keras

if keras.backend.backend() == "torch":
    import torch
    print("Use torch backend")
    torch.autograd.set_grad_enabled(False)

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "../"))
sys.path.append(os.path.join(current_dir, "../../"))

import bayesflow as bf
from torch import Tensor
from torch.distributions import Distribution
import torch.nn as nn

from custom_simulators import MyGenericSimulator, LikelihoodBasedModel, ParameterMask, Prior, RandomNumObs
from design_networks import RandomDesign, DeepAdaptiveDesign

class PolynomialRegression(LikelihoodBasedModel):
    def __init__(self, mask_sampler, prior_sampler, tau_sampler, design_generator, simulator_var) -> None:
        super().__init__(mask_sampler, prior_sampler, tau_sampler, design_generator, simulator_var)

    def outcome_likelihood(self, params: Tensor, xi: Tensor, simulator_var: dict) -> Distribution:

        xi_powers = torch.stack([torch.ones_like(xi), xi, xi ** 2, xi ** 3], dim=1) # [B, 4]
        mean = torch.sum(params * xi_powers, dim=-1, keepdim=True) # [B]
        sigma = simulator_var["sigma"]
        return torch.distributions.Normal(mean, sigma)
    
    def analytical_log_marginal_likelihood(outcomes, params: Tensor, masks: Tensor) -> Tensor:
        raise NotImplementedError # TODO
    

class PriorPolynomialReg(Prior):
    def __init__(self, delta: Tensor = Tensor([0.1])) -> None:
        super().__init__()
        self.delta = delta

    def dist(self, masks: Tensor) -> [Distribution]:
        super().__init__()
        
        self.masks = masks

        default = Tensor([[0, self.delta]])
        masks_ = masks.unsqueeze(-1)

        prior_0 = torch.where(masks_[:, 0] == 1, Tensor([5, 2]), default)
        prior_1 = torch.where(masks_[:, 1] == 1, Tensor([3, 1]), default)
        prior_2 = torch.where(masks_[:, 2] == 1, Tensor([0, 0.8]), default)
        prior_3 = torch.where(masks_[:, 3] == 1, Tensor([0, 0.5]), default)

        hyper_params = torch.stack([prior_0, prior_1, prior_2, prior_3], dim=1)

        means = hyper_params[:, :, 0]
        sds = hyper_params[:, :, 1]
    
        dist = torch.distributions.MultivariateNormal(means, scale_tril=torch.stack([torch.diag(sd) for sd in sds]))

        return dist
    

B = 64
batch_size = torch.Size([B])

mask_sampler = ParameterMask()
masks = mask_sampler(batch_size)
prior_sampler = PriorPolynomialReg()
params = prior_sampler.sample(masks)
likelihood = prior_sampler.log_prob(params, masks)


print(f"Shape of parameter mask {masks.shape}") # [B, model_dim]
print(f"Shape of parameters {params.shape}") # [B, param_dim]
print(f"Shape of likelihood {likelihood.shape}") # [B]

random_design_generator = RandomDesign()

T = 10

random_num_obs = RandomNumObs(min_obs = 1, max_obs = T)

polynomial_reg = PolynomialRegression(mask_sampler = mask_sampler,
                                      prior_sampler = prior_sampler,
                                      tau_sampler = random_num_obs,
                                      design_generator = random_design_generator,
                                      simulator_var = {"sigma": 1.0})

class MyDataSet(keras.utils.PyDataset):
    def __init__(self, batch_size: torch.Size, stage: int, initial_generative_model: MyGenericSimulator, design_network: nn.Module = None):
        super().__init__()

        self.batch_size = batch_size
        self.stage = stage # stage 1,2,3
        self.initial_generative_model = initial_generative_model
        self.design_network = design_network

    def __getitem__(self, item:int) -> dict[str, Tensor]:
        if self.stage == 1:

            data = self.initial_generative_model.sample(self.batch_size)
            return data

        if self.stage == 2:
            second_generative_model = 1
            data = self.second_generative_model.sampel(self.batch_size, )
            return data

        if self.stage == 3:
            ...
            return data
    
    @property
    def num_batches(self):
        # infinite dataset
        return None
    
dataset = MyDataSet(batch_size = batch_size, stage = 1, initial_generative_model = polynomial_reg)

# inference_network = bf.networks.CouplingFlow(depth = 8, subnet_kwargs=dict(kernel_regularizer=None, dropout_prob = False))
inference_network = bf.networks.FlowMatching()
summary_network = bf.networks.DeepSet(summary_dim = 10)

approximator = bf.Approximator(
    inference_network = inference_network,
    summary_network = summary_network,
    inference_variables = ["params"],
    inference_conditions = ["masks", "n_obs"],
    summary_variables = ["outcomes", "designs"]
)

approximator.compile(optimizer="AdamW")
approximator.fit(dataset, epochs=1, steps_per_epoch=1)

history = polynomial_reg.sample(torch.Size([1]))
ml = polynomial_reg.approximate_log_marginal_likelihood(100, history, approximator)