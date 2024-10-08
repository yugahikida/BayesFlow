import bayesflow as bf
import torch.nn as nn
from tqdm import trange
from torch.optim import Adam
import keras
import torch
import numpy as np
import os
import pickle
from keras.callbacks import History
import time

class JointApproximator:
    def __init__(self, 
                 approximator: bf.approximators, 
                 design_loss: nn.Module, 
                 dataset: keras.utils.PyDataset,
                 path_design_weight: str = None,
                 path_bf_weight: str = None) -> None:
        
        self.approximator = approximator
        self.design_loss = design_loss
        self.dataset = dataset
        self.path_design_weight = path_design_weight
        self.path_bf_weight = path_bf_weight

    def train(self, PATH: str, **hyper_params):
        PATH = os.path.join("DAD/results", PATH)
        if not os.path.exists(PATH):
            os.makedirs(PATH)

        # Stage 1: Train Bayesflow, use random design
        epochs_1 = hyper_params["epochs_1"]
        steps_per_epoch_1 = hyper_params["steps_per_epoch_1"]
        self.dataset.set_stage(1)


        init_lr = 1e-5
        warmup_target_lr = 1e-4
        learning_rate = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate = init_lr,
            decay_steps = epochs_1,
            alpha = 0.01,
            warmup_target = warmup_target_lr,
            warmup_steps = int(0.1 * epochs_1),
        )
        optimizer_bf = keras.optimizers.Adam(
            learning_rate = learning_rate,
            clipnorm = 1.0,
        )

        self.approximator.compile(optimizer=optimizer_bf)
        print("Stage 1: Train Bayesflow, use random design")
        PATH_1 = os.path.join(PATH, "approximator.weights.h5")

        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            PATH_1,
            verbose=0,
            save_best_only=False,
            save_weights_only=True
        )
        hist = History()
        if self.path_bf_weight is not None:
            self.approximator.build_from_data(self.dataset[0])
            self.approximator.load_weights(self.path_bf_weight)

        self.approximator.fit(dataset = self.dataset, epochs = epochs_1, steps_per_epoch = steps_per_epoch_1, 
                              callbacks=[model_checkpoint_callback, hist])
        self.approximator.save_weights(os.path.join(PATH, "approximator_stage_1.weights.h5"))
        PATH_approx_opt = os.path.join(PATH, 'approximator_optimizer_config.pkl')
        with open(PATH_approx_opt, 'wb') as file:
            pickle.dump(self.approximator.optimizer.get_config(), file)

        # Stage 2: Fix BayesFlow, train design network
        if self.path_design_weight is not None:
            self.design_loss.joint_model.design_generator.load_state_dict(torch.load(self.path_design_weight)["model_state_dict"])

        self.approximator.summary_network.trainable = False
        self.approximator.inference_network.trainable = False
        losses_design = []; taus = []; test_sims_list = []
        epochs_2 = hyper_params["epochs_2"]; steps_per_epoch_2 = hyper_params["steps_per_epoch_2"]
        # n_history = hyper_params["n_history"]
        policy_net = self.design_loss.joint_model.design_generator
        trainable_params = [param for param in policy_net.parameters() if param.requires_grad]
        optimizer = Adam(trainable_params, lr=1e-4)
        clipping_value = 1.0
        PATH_2 = os.path.join(PATH, "design_network.pt")

        # calculate scale
        scale_list = []
        n_sim_scale = 10
        for tau in range(0, self.design_loss.joint_model.tau_sampler.max_obs + 1):
            scale_tau = 0
            for _ in range(n_sim_scale):
                history = self.design_loss.simulate_history(n_history = 1, tau = tau)
                loss = self.design_loss(history = history) if tau != 0 else self.design_loss(history = None)
                scale_tau += loss.item()
            scale_list.append(scale_tau / n_sim_scale)
        scale_list = [abs(x) for x in scale_list]
        # scale_list = np.array(scale_list) - scale_list[0]

        loss_by_tau = np.zeros(self.design_loss.joint_model.tau_sampler.max_obs + 1)
        n_tau = np.zeros(self.design_loss.joint_model.tau_sampler.max_obs + 1)

        print("Stage 2: Fix BayesFlow weights, train design network")
        with torch.enable_grad():
            for e in range(epochs_2):
                print(f"Epoch {e + 1} / {epochs_2}")
                pbar = trange(steps_per_epoch_2)
                for _ in pbar:
                    optimizer.zero_grad()
                    tau = self.design_loss.joint_model.tau_sampler()
                    history = self.design_loss.simulate_history(n_history = 1, tau = tau)
                    loss = self.design_loss(history)
                    loss = loss / scale_list[tau] * 100
                    loss.backward()
                    torch.nn.utils.clip_grad_norm(policy_net.parameters(), clipping_value)
                    optimizer.step()
                    losses_design.append(loss.item()); taus.append(tau)
                    loss_by_tau[tau] += loss.item()
                    n_tau[tau] += 1
                    pbar.set_description(f"Loss: {loss.item():.4f}")
                #test_sims_list.append(self.design_loss.joint_model.sample(torch.Size([2]), tau = torch.tensor([5])))
                print(f"Loss: {round(np.mean(losses_design[-steps_per_epoch_2:]), 4)}")
                torch.save({'model_state_dict': self.design_loss.joint_model.design_generator.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()}, 
                            PATH_2)
        torch.save({'model_state_dict': self.design_loss.joint_model.design_generator.state_dict()}, os.path.join(PATH, "design_network_stage_2.pt"))
        np.savetxt(os.path.join(PATH, 'loss_by_tau.txt'), loss_by_tau / n_tau, fmt='%.3f')

        # Stage 3: Joint training
        self.approximator.summary_network.trainable = True  # Unfreeze weight for summary network
        self.approximator.inference_network.trainable = True  
        self.dataset.set_stage(3)
        epochs_3 = hyper_params["epochs_3"]
        steps_per_epoch_3 = hyper_params["steps_per_epoch_3"]

        print("Stage 3: Joint training")
        for e in range(epochs_3):
            print(f"Epoch {e + 1} / {epochs_3}")
            # bf
            self.approximator.load_weights(PATH_1)
            self.approximator.optimizer.from_config(pickle.load(open(PATH_approx_opt, 'rb'))) # load optimizer config
            self.approximator.fit(dataset = self.dataset, epochs = epochs_3, steps_per_epoch = steps_per_epoch_3, 
                              callbacks=[model_checkpoint_callback, hist])
            with open(PATH_approx_opt, 'wb') as file:
                pickle.dump(self.approximator.optimizer.get_config(), file)  # save optimizer config
            # design
            self.design_loss.joint_model.design_generator.load_state_dict(torch.load(PATH_2)["model_state_dict"])
            optimizer.load_state_dict(torch.load(PATH_2)["optimizer_state_dict"])
            pbar = trange(steps_per_epoch_3)
            for _ in pbar:
                with torch.enable_grad():
                    optimizer.zero_grad()
                    tau = self.design_loss.joint_model.tau_sampler()
                    history = self.design_loss.simulate_history(n_history = 1, tau = tau)
                    loss = self.design_loss(history)
                    loss.backward()
                    optimizer.step()
                    taus.append(tau); losses_design.append(loss.item())
                    pbar.set_description(f"Loss: {loss.item():.4f}")
            print(f"Loss: {round(np.mean(losses_design), 4)}")
            torch.save({'model_state_dict': self.design_loss.joint_model.design_generator.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}, PATH_2)
            
        with open(os.path.join(PATH, 'bf_loss.pkl'), 'wb') as f:
            pickle.dump(hist.history, f)
        with open(os.path.join(PATH, 'design_loss.pkl'), 'wb') as f:
            pickle.dump(losses_design, f)
        with open(os.path.join(PATH, 'tau.pkl'), 'wb') as f:
            pickle.dump(taus, f)
        with open(os.path.join(PATH, 'test_sims_transition.pkl'), 'wb') as f:
            pickle.dump(test_sims_list, f)
            

class DesignApproximator:
    def __init__(self, 
                 approximator: bf.approximators, 
                 design_loss: nn.Module, 
                 dataset: keras.utils.PyDataset) -> None:
        
        self.approximator = approximator
        self.design_loss = design_loss
        self.dataset = dataset

    def train(self, PATH: str, **hyper_params):
        PATH = os.path.join("DAD/results", PATH)
        
        if not os.path.exists(PATH):
            os.makedirs(PATH)

        # Stage 1: Train design network using prior predictive samples (DAD)
        epochs_1 = hyper_params["epochs_1"]
        steps_per_epoch_1 = hyper_params["steps_per_epoch_1"]
        PATH_1 = os.path.join(PATH, "design_network.pt")
        losses_design = []; taus = []
        trainable_params = [param for param in self.design_loss.joint_model.design_generator.parameters() if param.requires_grad]
        optimizer = Adam(trainable_params, lr=0.01)

        start_time = time.time()

        print("Stage 1: Train design network using prior predictive samples")
        with torch.enable_grad():
            for e in range(epochs_1):
                print(f"Epoch {e + 1} / {epochs_1}")
                pbar = trange(steps_per_epoch_1)
                for _ in pbar:
                    optimizer.zero_grad()
                    loss = self.design_loss(history = None)
                    loss.backward()
                    optimizer.step()
                    losses_design.append(loss.item())
                    pbar.set_description(f"Loss: {loss.item():.4f}")
                # test_sims_list.append(self.design_loss.joint_model.sample(torch.Size([1]), tau = torch.tensor([5]), masks = torch.tensor(mask), params = prior_samples))
                print(f"Loss: {round(np.mean(losses_design[-steps_per_epoch_1:]), 4)}")
                torch.save({'model_state_dict': self.design_loss.joint_model.design_generator.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()}, PATH_1)
        torch.save({'model_state_dict': self.design_loss.joint_model.design_generator.state_dict()}, os.path.join(PATH, "design_network_stage_1.pt"))

        duration = time.time() - start_time
        print(f"Training time: {duration}")
        with open(os.path.join(PATH, "time.txt"), 'a') as f: 
            f.write(str(duration))

        self.dataset.set_stage(3) # tau does not take zero
        init_lr = 1e-5
        warmup_target_lr = 1e-4
        learning_rate = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate = init_lr,
            decay_steps = epochs_1,
            alpha = 0.01,
            warmup_target = warmup_target_lr,
            warmup_steps = int(0.1 * epochs_1),
        )
        optimizer_bf = keras.optimizers.Adam(
            learning_rate = learning_rate,
            clipnorm = 1.0,
        )
        self.approximator.compile(optimizer=optimizer_bf)
        PATH_2 = os.path.join(PATH, "approximator.weights.h5")
        epochs_2 = hyper_params["epochs_2"]; steps_per_epoch_2 = hyper_params["steps_per_epoch_2"]

        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            PATH_2,
            verbose=0,
            save_best_only=False,
            save_weights_only=True
        )
        hist = History()
        print("Stage 2: Train Bayesflow with policy network")
        self.approximator.fit(dataset = self.dataset, epochs = epochs_2, steps_per_epoch = steps_per_epoch_2, 
                              callbacks=[model_checkpoint_callback, hist])
        self.approximator.save_weights(os.path.join(PATH, "approximator_stage_2.weights.h5"))
        with open(os.path.join(PATH, 'bf_loss.pkl'), 'wb') as f:
            pickle.dump(hist.history, f)
        with open(os.path.join(PATH, 'design_loss.pkl'), 'wb') as f:
            pickle.dump(losses_design, f)
        with open(os.path.join(PATH, 'tau.pkl'), 'wb') as f:
            pickle.dump(taus, f)