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

from custom_simulators import LikelihoodBasedModel, ParameterMask, Prior, RandomNumObs
from design_networks import RandomDesign, DeepAdaptiveDesign, EmitterNetwork
from design_loss import NestedMonteCarlo
from inference_design_approximator import InferenceDesignApproximator
from custom_dataset import MyDataSet


class PolynomialRegression(LikelihoodBasedModel):
    def __init__(self, mask_sampler, prior_sampler, tau_sampler, design_generator, simulator_var) -> None:
        super().__init__(mask_sampler, prior_sampler, tau_sampler, design_generator, simulator_var)

    def outcome_likelihood(self, params: Tensor, xi: Tensor, simulator_var: dict) -> Distribution:
        xi_powers = torch.stack([torch.ones_like(xi), xi, xi ** 2, xi ** 3], dim=2).squeeze(-1) # [B, 1, 4]
        mean = torch.sum(params.unsqueeze(1) * xi_powers, dim=-1, keepdim=True) # sum([B, 1, 4] * [B, 1, 4]) = [B, 1, y_dim] (y_dim = 1 here)
        sigma = simulator_var["sigma"]
        return torch.distributions.Normal(mean, sigma) # [B, 1, y_dim]
    
    def analytical_log_marginal_likelihood(outcomes, params: Tensor, masks: Tensor) -> Tensor:
        raise NotImplementedError # TODO
    

class PriorPolynomialReg(Prior):
    def __init__(self, delta: Tensor = Tensor([0.1])) -> None:
        super().__init__()
        self.delta = delta

    def dist(self, masks: Tensor) -> Distribution:
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
    
        dist = torch.distributions.MultivariateNormal(means, scale_tril=torch.stack([torch.diag(sd) for sd in sds])) # [B, theta_dim]

        return dist

inference_network = bf.networks.FlowMatching()
summary_network = bf.networks.DeepSet(summary_dim = 10)
approximator = bf.Approximator(
    inference_network = inference_network,
    summary_network = summary_network,
    inference_variables = ["params"],
    inference_conditions = ["masks", "n_obs"],
    summary_variables = ["outcomes", "designs"]
)

T = 10 
design_shape = torch.Size([1])
mask_sampler = ParameterMask()
prior_sampler = PriorPolynomialReg()
random_num_obs = RandomNumObs(min_obs = 1, max_obs = T)
random_design_generator = RandomDesign(design_shape = design_shape)

polynomial_reg_1 = PolynomialRegression(mask_sampler = mask_sampler,
                                        prior_sampler = prior_sampler,
                                        tau_sampler = random_num_obs,
                                        design_generator = random_design_generator,
                                        simulator_var = {"sigma": 1.0})


decoder_net = EmitterNetwork(input_dim = 10, hidden_dim = 24, output_dim = 1) # [B, summary_dim] -> [B, design_dim]
design_net = DeepAdaptiveDesign(encoder_net = approximator.summary_network,
                                decoder_net = decoder_net,
                                design_shape = design_shape, 
                                summary_variables=["outcomes", "designs"])

polynomial_reg_2 = PolynomialRegression(mask_sampler = mask_sampler,
                                        prior_sampler = prior_sampler,
                                        tau_sampler = random_num_obs,
                                        design_generator = design_net,
                                        simulator_var = {"sigma": 1.0})

# hyperparams for bf
B = 256
batch_shape_b = torch.Size([B])

# hyperparams for DAD
batch_shape_d = torch.Size([B])
L = 256

pce_loss = NestedMonteCarlo(joint_model = polynomial_reg_2,
                            approximator = approximator,
                            batch_shape = batch_shape_d,
                            num_negative_samples = L)

dataset = MyDataSet(batch_shape = batch_shape_b, 
                    joint_model_1 = polynomial_reg_1,
                    joint_model_2 = polynomial_reg_2)

trainer = InferenceDesignApproximator(
    approximator = approximator,
    design_loss = pce_loss,
    dataset = dataset
)

dataset.set_stage(1)
approximator.compile(optimizer="AdamW")
approximator.fit(dataset, epochs=10, steps_per_epoch = 10)

hyper_params = {"num_steps_1": 5000, "num_steps_2": 5000, "num_steps_3": 500}


trainer.train(PATH = "test", **hyper_params)