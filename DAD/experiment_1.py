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
from design_networks import RandomDesign, DeepAdaptiveDesign, EmitterNetwork, EncoderNetwork
from design_loss import NestedMonteCarlo
from inference_design_approximator import InferenceDesignApproximator
from custom_dataset import MyDataSet

import argparse

class PolynomialRegression(LikelihoodBasedModel):
    def __init__(self, mask_sampler, prior_sampler, tau_sampler, design_generator, simulator_var) -> None:
        super().__init__(mask_sampler, prior_sampler, tau_sampler, design_generator, simulator_var)

    def outcome_likelihood(self, params: Tensor, xi: Tensor, simulator_var: dict) -> Distribution: # params: [B, param_dim], xi: [B, 1, xi_dim]
        xi_powers = torch.stack([torch.ones_like(xi), xi, xi ** 2, xi ** 3], dim=-2).squeeze(-1) # [B, 1, 4]
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
    
def experiment_1(PATH: str = "test",
                 epochs_1: int = 1,
                 epochs_2: int = 1,
                 epochs_3: int = 1,
                 steps_per_epoch_1: int = 50,
                 steps_per_epoch_2: int = 50,
                 steps_per_epoch_3: int = 50,
                 bf_summary_dim: int = 10, 
                 T: int = 20,
                 noize_size: float = 1.0, 
                 scaler: int = 5,
                 dad_encoder_hidden_dim: int = 32, 
                 dad_summary_dim: int = 10,
                 dad_emitter_hidden_dim: int = 2,
                 bf_batch_size: int = 128,
                 dad_positive_samples: int = 2000,
                 dad_negative_samples: int = 2000,
                 ) -> None:
    inference_network = bf.networks.FlowMatching() # bf.networks.CouplingFlow()
    summary_network = bf.networks.DeepSet(summary_dim = bf_summary_dim)

    approximator = bf.Approximator(
        inference_network = inference_network,
        summary_network = summary_network,
        inference_variables = ["params"],
        inference_conditions = ["masks", "n_obs"],
        summary_variables = ["outcomes", "designs"]
    )
    design_shape = torch.Size([1])
    mask_sampler = ParameterMask()
    prior_sampler = PriorPolynomialReg()
    random_num_obs = RandomNumObs(min_obs = 1, max_obs = T)
    random_design_generator = RandomDesign(design_shape = design_shape, scaler = scaler)

    polynomial_reg_1 = PolynomialRegression(mask_sampler = mask_sampler,
                                            prior_sampler = prior_sampler,
                                            tau_sampler = random_num_obs,
                                            design_generator = random_design_generator,
                                            simulator_var = {"sigma": noize_size})

    encoder_net = EncoderNetwork(xi_dim = 1, y_dim = 1, hidden_dim = dad_encoder_hidden_dim, encoding_dim = dad_summary_dim)
    decoder_net = EmitterNetwork(input_dim = dad_summary_dim, hidden_dim = dad_emitter_hidden_dim, output_dim = 1, scaler = scaler) # [B, summary_dim] -> [B, xi_dim]
    design_net = DeepAdaptiveDesign(encoder_net = encoder_net,
                                    decoder_net = decoder_net,
                                    design_shape = design_shape, 
                                    summary_variables=["outcomes", "designs"])

    polynomial_reg_2 = PolynomialRegression(mask_sampler = mask_sampler,
                                            prior_sampler = prior_sampler,
                                            tau_sampler = random_num_obs,
                                            design_generator = design_net,
                                            simulator_var = {"sigma": noize_size})

    batch_shape_b = torch.Size([bf_batch_size])
    batch_shape_d = torch.Size([dad_positive_samples])

    dataset = MyDataSet(batch_shape = batch_shape_b, 
                        joint_model_1 = polynomial_reg_1,
                        joint_model_2 = polynomial_reg_2)

    pce_loss = NestedMonteCarlo(approximator = approximator,
                                joint_model = polynomial_reg_2, # joint model with design network
                                batch_shape = batch_shape_d,
                                num_negative_samples = dad_negative_samples)

    trainer = InferenceDesignApproximator(
        approximator = approximator,
        design_loss = pce_loss,
        dataset = dataset
    )

    hyper_params = {"epochs_1": epochs_1, "steps_per_epoch_1": steps_per_epoch_1,
                    "epochs_2": epochs_2, "steps_per_epoch_2": steps_per_epoch_2,
                    "epochs_3": epochs_3, "steps_per_epoch_3": steps_per_epoch_3}

    trainer.train(PATH = PATH, **hyper_params)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-path", type=str, default="test", help="path to save the results")
    parser.add_argument("-epochs_1", type=int, default=1, help="epochs for stage 1 (default=1)")
    parser.add_argument("-epochs_2", type=int, default=1, help="epochs for stage 2 (default=1)")
    parser.add_argument("-epochs_3", type=int, default=1, help="epochs for stage 3 (default=1)")
    parser.add_argument("-steps_per_epoch_1", type=int, default=50, help="steps per epoch for stage 1 (default=50)")
    parser.add_argument("-steps_per_epoch_2", type=int, default=50, help="steps per epoch for stage 2 (default=50)")
    parser.add_argument("-steps_per_epoch_3", type=int, default=50, help="steps per epoch for stage 3 (default=50)")
    parser.add_argument("-bf_summary_dim", type=int, default=10, help="summary dimension for bf (default=10)")
    parser.add_argument("-T", type=int, default=20, help="maximum number of observations (default=20)")
    parser.add_argument("-noize_size", type=float, default=1.0, help="size of noise (default=1.0)")
    parser.add_argument("-scaler", type=int, default=5, help="range that xi can take [-scaler, scaler] (default=5)")
    parser.add_argument("-dad_encoder_hidden_dim", type=int, default=32, help="hidden dimension for encoder network in dad (default=32)")
    parser.add_argument("-dad_summary_dim", type=int, default=10, help="summary dimension for dad (default=10)")
    parser.add_argument("-dad_emitter_hidden_dim", type=int, default=2, help="hidden dimension for emitter network in dad (default=2)")
    parser.add_argument("-bf_batch_size", type=int, default=128, help="batch size for bf (default=128)")
    parser.add_argument("-dad_positive_samples", type=int, default=2000, help="number of positive samples for dad (default=2000)")
    parser.add_argument("-dad_negative_samples", type=int, default=2000, help="number of negative samples for dad (default=2000)")
    args = parser.parse_args()
    print(args)


    PATH = os.path.join("DAD", args.path)
        
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    with open(os.path.join(PATH, 'arguments.txt'), 'w') as file:
        for arg, value in vars(args).items():
            file.write(f'{arg}: {value}\n')

    experiment_1(PATH = args.path,
                 epochs_1 = args.epochs_1,
                 epochs_2 = args.epochs_2,
                 epochs_3 = args.epochs_3,
                 steps_per_epoch_1 = args.steps_per_epoch_1,
                 steps_per_epoch_2 = args.steps_per_epoch_2,
                 steps_per_epoch_3 = args.steps_per_epoch_3,
                 bf_summary_dim = args.bf_summary_dim,
                 T = args.T,
                 noize_size = args.noize_size,
                 dad_encoder_hidden_dim = args.dad_encoder_hidden_dim,
                 dad_summary_dim = args.dad_summary_dim,
                 dad_emitter_hidden_dim = args.dad_emitter_hidden_dim,
                 bf_batch_size = args.bf_batch_size,
                 dad_positive_samples = args.dad_positive_samples,
                 dad_negative_samples = args.dad_negative_samples)