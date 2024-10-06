import os, pickle
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
    
def evaluate_loss(PATH, tau_eval=None):
    """
    PATH: path of the folder containing the output files
    tau_eval: list of tau values to evaluate the loss by tau. If None, all tau values are considered.
    """

    with (open(os.path.join(PATH, 'tau.pkl'), 'rb') as f1,
          open(os.path.join(PATH, 'design_loss.pkl'), 'rb') as f2,
          open(os.path.join(PATH, 'arguments.pkl'), 'rb') as f3,
          open(os.path.join(PATH, 'bf_loss.pkl'), 'rb') as f4):
        
        tau = pickle.load(f1)
        design_loss = pickle.load(f2)
        args_dict = pickle.load(f3)
        bf_loss = pickle.load(f4)["loss"]
   
    n_steps_1 = args_dict["epochs_1"] * args_dict["steps_per_epoch_1"]
    n_steps_2 = args_dict["epochs_2"] * args_dict["steps_per_epoch_2"]
    n_steps_3 = args_dict["epochs_3"] * args_dict["steps_per_epoch_3"]

    joint = args_dict["joint"]

    if joint == False:
        fig, axs = plt.subplots(2, figsize=(8, 6))
        axs[0].plot(range(n_steps_1), design_loss[0:n_steps_1], color = "#791F1F")
        axs[0].set_title("Loss for Policy network")
        axs[0].set_xlabel('Steps')
        axs[0].set_ylabel('Loss value')

        axs[1].plot(range(args_dict["epochs_2"]), bf_loss[0:args_dict["epochs_2"]], color = "#791F1F")
        axs[1].set_title("Loss for BayesFlow")
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Loss value')

        plt.tight_layout(pad=3.0)

        return fig
        
    else:
        fig = plt.figure(constrained_layout=True, figsize= (12, 8))
        gs = GridSpec(4, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[2, 0])
        ax4 = fig.add_subplot(gs[3, 0])
        ax5 = fig.add_subplot(gs[0:2, 1])
        ax6 = fig.add_subplot(gs[2:4, 1])

        ax1.plot(range(args_dict["epochs_1"]), bf_loss[0:args_dict["epochs_1"]], color = "#791F1F")
        ax1.set_title("Loss for BayesFlow (Stage 1)")
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss value')

        ax2.plot(range(n_steps_2), design_loss[0:n_steps_2], color = "#791F1F")
        ax2.set_title("Loss for Policy network (Stage 2)")
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Loss value')

        ax3.plot(range(args_dict["epochs_3"]), bf_loss[args_dict["epochs_1"]:], color = "#791F1F")
        ax3.set_title("Loss for BayesFlow (Stage 3)")
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Loss value')

        ax4.plot(range(n_steps_3), design_loss[n_steps_2:], color = "#791F1F")
        ax4.set_title("Loss for Policy network (Stage 3)")
        ax4.set_xlabel('Steps')
        ax4.set_ylabel('Loss value')

        if tau_eval is None:
            tau_eval = range(args_dict["T"])

        for t in tau_eval:
            loss_tau = [design_loss[0:n_steps_2][i] for i in range(n_steps_2) if tau[0:n_steps_2][i] == t]
            ax5.plot(range(len(loss_tau)), loss_tau, label=f"tau={t}")
        ax5.legend()
        ax5.set_title("Loss for Policy network by tau (Stage 2)")

        for t in tau_eval:
            loss_tau = [design_loss[n_steps_2:][i] for i in range(n_steps_3) if tau[n_steps_2:][i] == t]
            ax6.plot(range(len(loss_tau)), loss_tau, label=f"tau={t}")
        ax6.legend()
        ax6.set_title("Loss for Policy network by tau (Stage 3)")
        
        return gs
    
# import os
# os.environ["KERAS_BACKEND"] = "torch"
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# import keras

# if keras.backend.backend() == "torch":
#     import torch
#     print("Use torch backend")
#     torch.autograd.set_grad_enabled(False)

# import sys

# current_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.join(current_dir, "../"))
# sys.path.append(os.path.join(current_dir, "../../"))

# import bayesflow as bf

# from custom_simulators import ParameterMask, RandomNumObs
# from design_networks import RandomDesign, DeepAdaptiveDesign, EncoderNetwork, EmitterNetwork, DeepAdaptiveDesignSimple
# # from design_networks_old import EmitterNetwork as EmitterNetwork_old
# from custom_dataset import DataSet
# from experiment_1 import PolynomialRegression, PriorPolynomialReg, PolynomialRegressionPoisson
# import pickle
# import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec


# def get_data_for_plot_recovery(path, prior_samples, masks, n_sims = 2000, n_samples = 100, random = False): # TODO: delete old_emit later on we don't need

#     with open(os.path.join(path, 'arguments.pkl'), 'rb') as file:
#         args_dict = pickle.load(file)

#     inference_network = bf.networks.FlowMatching() # bf.networks.CouplingFlow()
#     summary_network = bf.networks.DeepSet(summary_dim = args_dict["bf_summary_dim"])

#     approximator = bf.approximators(
#         inference_network = inference_network,
#         summary_network = summary_network,
#         inference_variables = ["params"],
#         inference_conditions = ["masks", "n_obs"],
#         summary_variables = ["outcomes", "designs"]
#     )

#     # mask = [[1., 1., 1., 1.]]
#     design_shape = torch.Size([1])
#     mask_sampler = ParameterMask(possible_masks = torch.tensor(masks))
#     prior_sampler = PriorPolynomialReg()
#     random_num_obs_1 = RandomNumObs(min_obs = 1, max_obs = args_dict["T"])
#     random_num_obs_2 = RandomNumObs(min_obs = 0, max_obs = args_dict["T"])


#     # encoder_net = EncoderNetwork(xi_dim = 1, y_dim = 1, hidden_dim = args_dict["dad_encoder_hidden_dim"], encoding_dim = args_dict["dad_summary_dim"])


#     random_design_generator = RandomDesign(design_shape = design_shape, min = args_dict["min_design"], max = args_dict["max_design"])
#     # decoder_net = EmitterNetwork(input_dim = args_dict["dad_summary_dim"], hidden_dim = args_dict["dad_emitter_hidden_dim"], output_dim = 1, min = args_dict["min_design"], max = args_dict["max_design"]) # [B, summary_dim] -> [B, xi_dim]


#     # design_net = DeepAdaptiveDesign(encoder_net = encoder_net,
#     #                                 decoder_net = decoder_net,
#     #                                 design_shape = design_shape, 
#     #                                 summary_variables=["outcomes", "designs"])

#     design_net = DeepAdaptiveDesignSimple(design_shape = 1,
#                                           y_dim = 1,
#                                           hidden_dim = args_dict["dad_emitter_hidden_dim"],
#                                           embedding_dim = args_dict["dad_encoder_hidden_dim"],
#                                           batch_size = args_dict["dad_summary_dim"])
    
#     model_1 = PolynomialRegressionPoisson(mask_sampler = mask_sampler,
#                                                    prior_sampler = prior_sampler,
#                                                    tau_sampler = random_num_obs_1,
#                                                    design_generator = random_design_generator,
#                                                    sim_vars ={})
    
#     model_2 = PolynomialRegressionPoisson(mask_sampler = mask_sampler,
#                                                    prior_sampler = prior_sampler,
#                                                    tau_sampler = random_num_obs_2,
#                                                    design_generator = design_net,
#                                                    sim_vars = {})
    
#     model_3 = PolynomialRegressionPoisson(mask_sampler = mask_sampler,
#                                                    prior_sampler = prior_sampler,
#                                                    tau_sampler = random_num_obs_1,
#                                                    design_generator = design_net,
#                                                    sim_vars = {}) # for recovery

#     batch_shape_b = torch.Size([args_dict["bf_batch_size"]])

#     dataset = DataSet(batch_shape = batch_shape_b, 
#                             joint_model_1 = model_1,
#                             joint_model_2 = model_2,
#                             joint_model_3 = model_3)
    
#     dataset.set_stage(1)
#     approximator.build_from_data(dataset.__getitem__(0))
        
    
#     if args_dict["dad"] == "second":
#         path_design_approximator = path + "/design_network.pt"
#         state_dict = torch.load(path_design_approximator)["model_state_dict"]
#         model_2.design_generator.load_state_dict(state_dict)

#         if random:
#             path_bf_weights_s1 = path + "/approximator_stage_1.weights.h5"
#             approximator.load_weights(path_bf_weights_s1)
#             test_sims = model_1.sample(torch.Size([n_sims]), tau = torch.tensor([args_dict["T"]]), masks = torch.tensor(masks), params = prior_samples)
#             post_samples = approximator.sample((n_sims, n_samples), data = test_sims)["params"].to('cpu').numpy()
#             return test_sims, post_samples
        
#         else:
#             path_bf_weights_s3 = path + "/approximator.weights.h5"
#             approximator.load_weights(path_bf_weights_s3)
#             test_sims = model_2.sample(torch.Size([n_sims]), tau = torch.tensor([args_dict["T"]]), masks = torch.tensor(masks), params = prior_samples)
#             post_samples = approximator.sample((n_sims, n_samples), data = test_sims)["params"].to('cpu').numpy()
#             return test_sims, post_samples
#     else:
#         path_design_approximator = path + "/design_network_stage_1.pt"
#         state_dict = torch.load(path_design_approximator)["model_state_dict"]
#         model_2.design_generator.load_state_dict(state_dict)
#         path_bf_weights_s2 = path + "/approximator_stage_2.weights.h5"
#         approximator.load_weights(path_bf_weights_s2)
#         test_sims = model_2.sample(torch.Size([n_sims]), tau = torch.tensor([args_dict["T"]]), masks = torch.tensor(masks), params = prior_samples)
#         post_samples = approximator.sample((n_sims, n_samples), data = test_sims)["params"].to('cpu').numpy()
#         return test_sims, post_samples