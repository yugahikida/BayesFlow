import warnings
warnings.filterwarnings('ignore')

#### bf settings
import os
os.environ["KERAS_BACKEND"] = "torch"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import keras

if keras.backend.backend() == "torch":
    import torch
    torch.autograd.set_grad_enabled(False)

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "../"))
sys.path.append(os.path.join(current_dir, "../../"))

import bayesflow as bf
from torch import Tensor
from torch.distributions import Distribution
import torch.distributions as dist

from custom_simulators import LikelihoodBasedModel, ParameterMask, Prior, RandomNumObs
from design_networks import RandomDesign, EmitterNetwork, EncoderNetwork, DADSimple, DADMulti, DADMulti2
from design_loss import NestedMonteCarlo
from inference_design_approximator import JointApproximator, DesignApproximator
from custom_dataset import DataSet
import pickle
import argparse

class PolyReg(LikelihoodBasedModel):
    def __init__(self, mask_sampler, prior_sampler, tau_sampler, design_generator, sim_vars) -> None:
        super().__init__(mask_sampler, prior_sampler, tau_sampler, design_generator, sim_vars)
        self.sim_vars = sim_vars

    def get_designs_matrix(self, designs: Tensor) -> Tensor:
        designs = torch.cat([designs**i for i in range(1, self.sim_vars["degree"] + 1)], dim=-1)

        if self.sim_vars["include_intercept"]:
            design_matrix = torch.cat(
                [torch.ones_like(designs[..., :1]), designs], dim=-1
            )
        else:
            design_matrix = designs
        return design_matrix
    
    def outcome_likelihood(self, params: Tensor, xi: Tensor) -> Distribution: 
        NotImplementedError

class StudentT(PolyReg):
    def __init__(self, mask_sampler, prior_sampler, tau_sampler, design_generator, sim_vars) -> None:
        super().__init__(mask_sampler, prior_sampler, tau_sampler, design_generator, sim_vars)
        self.sim_vars = sim_vars

    def outcome_likelihood(self, params: Tensor, xi: Tensor) -> Distribution: # params: [B, param_dim], xi: [B, 1, xi_dim]
        design_matrix = self.get_designs_matrix(xi)
        mean_outcome = torch.sum(design_matrix * params.unsqueeze(1), dim=-1, keepdim=True)  # sum([B, 1, param_dim] * [B, 1, param_dim]) = [B, 1, y_dim] (y_dim = 1 here)
        return dist.StudentT(df = 10, loc = mean_outcome, scale = self.sim_vars["noise_size"]) # [B, 1, y_dim]

class Expo(PolyReg):
    def __init__(self, mask_sampler, prior_sampler, tau_sampler, design_generator, sim_vars = None) -> None:
        super().__init__(mask_sampler, prior_sampler, tau_sampler, design_generator, sim_vars)

    def outcome_likelihood(self, params: Tensor, xi: Tensor) -> Distribution:
        design_matrix = self.get_designs_matrix(xi)
        mean_outcome = torch.sum(design_matrix * params.unsqueeze(1), dim=-1, keepdim=True)  
        return dist.Exponential(rate = mean_outcome)

class PriorPolynomialReg(Prior):
    def __init__(self, delta: Tensor = Tensor([1e-5])) -> None: # delta to be much smaller 
        super().__init__()
        self.delta = delta

    def dist(self, masks: Tensor) -> Distribution:
        sigma_prior = 2.0
        masks = masks * sigma_prior
        masks[masks == 0] = self.delta
        means = torch.zeros_like(masks)
        return dist.MultivariateNormal(means, scale_tril=torch.stack([torch.diag(mask) for mask in masks])) # [B, theta_dim]
    
def experiment_1(PATH: str = "test",
                 epochs_1: int = 1,
                 epochs_2: int = 1,
                 epochs_3: int = 1,
                 steps_per_epoch_1: int = 50,
                 steps_per_epoch_2: int = 50,
                 steps_per_epoch_3: int = 50,
                 dad_summary_dim: int = 10,
                 dad_encoder_hidden_dim: int = 64, 
                 dad_emitter_hidden_dim: int = 32,
                 dad_positive_samples: int = 256,
                 dad_negative_samples: int = 128,
                 joint: bool = False,
                 path_design_weight: str = None,
                 path_bf_weight: str = None,
                 degree: str = 1,
                 include_intercept: bool = True,
                 single_model: bool = True
                 ) -> None:
    
    # Fixed settings
    n_history = 1
    T = 10
    bf_summary_dim = 10
    bf_batch_size = 128
    design_size = 1

    inference_network = bf.networks.CouplingFlow(depth = 5)
    summary_network = bf.networks.DeepSet(summary_dim = bf_summary_dim)

    data_adapter = bf.ContinuousApproximator.build_data_adapter(
        inference_variables = ["params"],
        inference_conditions = ["masks", "n_obs"],
        summary_variables = ["outcomes", "designs"],
        transforms = None
    )

    approximator = bf.ContinuousApproximator(
            inference_network = inference_network,
            summary_network = summary_network,
            data_adapter = data_adapter
    )

    if include_intercept:
        possible_masks = torch.tril(torch.ones((degree + include_intercept, degree + include_intercept)))[1:, :]

    else:
        possible_masks = torch.tril(torch.ones((degree, degree)))

    if single_model:
        possible_masks = possible_masks[-1:, :] # only one model

    mask_sampler = ParameterMask(possible_masks = torch.tensor(possible_masks, dtype=torch.float32))
    prior_sampler = PriorPolynomialReg() # param_dim is define through possible_masks
    random_num_obs_1 = RandomNumObs(min_obs = 1, max_obs = T) # for bf
    random_num_obs_2 = RandomNumObs(min_obs = 0, max_obs = T - 1) # for dad
    sim_vars = {"degree": degree, "include_intercept": include_intercept, "noise_size": 1.0}

    random_design_generator = RandomDesign(design_size = design_size)
    model = StudentT
    model_1 = model(mask_sampler = mask_sampler,
                    prior_sampler = prior_sampler,
                    tau_sampler = random_num_obs_1,
                    design_generator = random_design_generator,
                    sim_vars = sim_vars)

    if single_model:
        design_net = DADSimple(design_size = 1,
                             y_dim = 1,
                             embedding_dim = dad_summary_dim,
                             batch_size = dad_positive_samples)

    else:
        design_net = DADMulti(design_size = 1,
                              y_dim = 1,
                              embedding_dim = dad_summary_dim,
                              context_dim = include_intercept + degree,
                              batch_size = dad_positive_samples,
                              T = T)
    
    model_2 = model(mask_sampler = mask_sampler,
                    prior_sampler = prior_sampler,
                    tau_sampler = random_num_obs_2,
                    design_generator = design_net,
                    sim_vars = sim_vars)
    
    model_3 = model(mask_sampler = mask_sampler,
                    prior_sampler = prior_sampler,
                    tau_sampler = random_num_obs_1,
                    design_generator = design_net,
                    sim_vars = sim_vars)

    dataset = DataSet(batch_size = bf_batch_size, 
                      joint_model_1 = model_1,
                      joint_model_2 = model_2,
                      joint_model_3 = model_3,
                      data_adapter = data_adapter)
    
    pce_loss = NestedMonteCarlo(approximator = approximator,
                                joint_model = model_2, # joint model with design network with tau taking 0.
                                batch_size = dad_positive_samples,
                                num_negative_samples = dad_negative_samples)
    
    eval_lower = NestedMonteCarlo(
        approximator = approximator,
        joint_model = model_2,
        batch_size = 1000,
        num_negative_samples = 10000,
        lower_bound = True,
    )

    eval_upper = NestedMonteCarlo(
        approximator = approximator,
        joint_model = model_2,
        batch_size = 1000,
        num_negative_samples = 10000,
        lower_bound = False,
    )

    # run 1 batch through the model
    _, _, _, designs, _ = model_2(batch_size=1, tau = T).values()
    print("\n")
    print(f"Initial designs={designs.reshape(-1)}")
    print(f"Lower bound={eval_lower.estimate():.4f}")
    print(f"Upper bound={eval_upper.estimate():.4f}")

    # joint = False
    if joint == True:
        trainer = JointApproximator(
            approximator = approximator,
            design_loss = pce_loss,
            dataset = dataset,
            path_design_weight = path_design_weight,
            path_bf_weight = path_bf_weight
        )

    else:
        trainer = DesignApproximator(
            approximator = approximator,
            design_loss = pce_loss,
            dataset = dataset
        )

    hyper_params = {"epochs_1": epochs_1, "steps_per_epoch_1": steps_per_epoch_1,
                    "epochs_2": epochs_2, "steps_per_epoch_2": steps_per_epoch_2,
                    "epochs_3": epochs_3, "steps_per_epoch_3": steps_per_epoch_3,
                    "n_history": n_history}

    trainer.train(PATH = PATH, **hyper_params)

    print("\n")
    print("EVALUATING")
    # run 1 batch through the model
    _, _, _, designs, _  = model_2(batch_size=1, tau = T).values()
    print(f"Lower bound={eval_lower.estimate():.4f}")
    print(f"Upper bound={eval_upper.estimate():.4f}")
    print(f"Final designs={designs.reshape(-1)}")

    # save samples from the final models
    data = model_2(batch_size=2000, tau = T)
    with open(os.path.join("DAD/results", PATH, 'data.pkl'), 'wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-path", type=str, default="test", help="path to save the results under results folder")
    parser.add_argument("-epochs_1", type=int, default=1)
    parser.add_argument("-epochs_2", type=int, default=1)
    parser.add_argument("-epochs_3", type=int, default=1)
    parser.add_argument("-steps_per_epoch_1", type=int, default=2)
    parser.add_argument("-steps_per_epoch_2", type=int, default=2)
    parser.add_argument("-steps_per_epoch_3", type=int, default=2)
    parser.add_argument("-dad_encoder_hidden_dim", type=int, default=64)
    parser.add_argument("-dad_summary_dim", type=int, default=10)
    parser.add_argument("-dad_emitter_hidden_dim", type=int, default=2)
    parser.add_argument("-dad_positive_samples", type=int, default=256)
    parser.add_argument("-dad_negative_samples", type=int, default=128)
    parser.add_argument("-joint", type=int, default=1, help="if joint training is used (default = True)")
    parser.add_argument("-path_design_weight", type=str, default=None, help="path to design network weights (default=None)")
    parser.add_argument("-path_bf_weight", type=str, default=None, help="path to bf weights (default=None)")
    parser.add_argument("-noise_size", type=float, default=1.0)
    parser.add_argument("-degree", type=int, default=1)
    parser.add_argument("-include_intercept", type=int, default=1) 
    parser.add_argument("-single_model", type=int, default=1, help="only estimate full model (default = True)")
    args = parser.parse_args()

    PATH = os.path.join("DAD/results", args.path)
        
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    import pickle

    with open(os.path.join(PATH, 'arguments.txt'), 'w') as f:
        for arg, value in vars(args).items():
            f.write(f'{arg}: {value}\n')

    with open(os.path.join(PATH, 'arguments.pkl'), 'wb') as f:
            pickle.dump(vars(args), f)

    experiment_1(PATH = args.path,
                 epochs_1 = args.epochs_1,
                 epochs_2 = args.epochs_2,
                 epochs_3 = args.epochs_3,
                 steps_per_epoch_1 = args.steps_per_epoch_1,
                 steps_per_epoch_2 = args.steps_per_epoch_2,
                 steps_per_epoch_3 = args.steps_per_epoch_3,
                 dad_encoder_hidden_dim = args.dad_encoder_hidden_dim,
                 dad_emitter_hidden_dim = args.dad_emitter_hidden_dim,
                 dad_positive_samples = args.dad_positive_samples,
                 dad_negative_samples = args.dad_negative_samples,
                 joint = bool(args.joint),
                 path_bf_weight= args.path_bf_weight,
                 path_design_weight= args.path_design_weight,
                 degree = args.degree,
                 include_intercept = bool(args.include_intercept),
                 single_model = bool(args.single_model),
                 dad_summary_dim = args.dad_summary_dim,
                 )
