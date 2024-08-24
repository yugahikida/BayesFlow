import bayesflow as bf
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
import keras
import torch
import os
import pickle
from keras.callbacks import History

class InferenceDesignApproximator:
    def __init__(self, 
                 approximator: bf.Approximator, 
                 design_loss: nn.Module, 
                 dataset: keras.utils.PyDataset) -> None:
        
        self.approximator = approximator
        self.design_loss = design_loss
        self.dataset = dataset

    def train(self, PATH: str, **hyper_params):
        PATH = os.path.join("DAD", PATH)
        
        if not os.path.exists(PATH):
            os.makedirs(PATH)

        # Stage 1: Train Bayesflow, use random design
        epochs_1 = hyper_params["epochs_1"]
        steps_per_epoch_1 = hyper_params["steps_per_epoch_1"]

        self.dataset.set_stage(1)
        self.approximator.compile(optimizer="AdamW")

        print("Stage 1: Train Bayesflow, use random design")

        PATH_1 = os.path.join(PATH, "approximator.weights.h5")

        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            PATH_1,
            verbose=0,
            save_best_only=False,
            save_weights_only=True
        )

        hist = History()

        self.approximator.fit(self.dataset, epochs = epochs_1, steps_per_epoch = steps_per_epoch_1, 
                              callbacks=[model_checkpoint_callback, hist])
        
        self.approximator.save_weights(os.path.join(PATH, "approximator_stage_1.weights.h5"))
        PATH_approx_opt = os.path.join(PATH, 'approximator_optimizer_config.pkl')
        with open(PATH_approx_opt, 'wb') as file:
            pickle.dump(self.approximator.optimizer.get_config(), file)

        # Stage 2: Fix BayesFlow, train design network
        self.approximator.summary_network.trainable = False  # Freeze weight for summary network

        losses_design = []
        taus = []

        epochs_2 = hyper_params["epochs_2"]
        steps_per_epoch_2 = hyper_params["steps_per_epoch_2"]

        trainable_params = [param for param in self.design_loss.joint_model.design_generator.parameters() if param.requires_grad]
        optimizer = Adam(trainable_params, lr=1e-3)
        PATH_2 = os.path.join(PATH, "design_network.pt")
        
        print("Stage 2: Fix BayesFlow weights, train design network")

        with torch.enable_grad():
            for e in range(epochs_2):
                print(f"Epoch {e + 1} / {epochs_2}")
                for _ in tqdm(range(steps_per_epoch_2)):
                    optimizer.zero_grad()

                    tau = self.design_loss.joint_model.tau_sampler()
                    taus.append(tau)
                    # taus.append(int(history["n_obs"].item()**2))
                    dad_batch_size = 32

                    history = self.design_loss.simulate_history(batch_size = dad_batch_size, tau = tau)
                    loss = self.design_loss(history)
                    
                    loss.backward()
                    optimizer.step()
                    losses_design.append(loss.item())

                print(f"Loss: {round(loss.item(), 4)}")

                torch.save({'model_state_dict': self.design_loss.joint_model.design_generator.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()}, 
                            PATH_2)
                
        torch.save({'model_state_dict': self.design_loss.joint_model.design_generator.state_dict()}, 
                    os.path.join(PATH, "design_network_stage_2.pt"))

        # Stage 3: Joint training
        self.approximator.summary_network.trainable = True  # Unfreeze weight for summary network
        self.dataset.set_stage(3)

        ## load weights and optimizer state from previous stages
        self.approximator.load_weights(PATH_1)
        self.approximator.optimizer.from_config(pickle.load(open(PATH_approx_opt, 'rb'))) 

        self.design_loss.joint_model.design_generator.load_state_dict(torch.load(PATH_2)["model_state_dict"])
        optimizer.load_state_dict(torch.load(PATH_2)["optimizer_state_dict"]) 
        
        epochs_3 = hyper_params["epochs_3"]
        steps_per_epoch_3 = hyper_params["steps_per_epoch_3"]

        print("Stage 3: Joint training")
        for e in range(epochs_3):
            print(f"Epoch {e + 1} / {epochs_3}")

            # bf
            self.approximator.load_weights(PATH_1) # load approximator weights
            self.approximator.optimizer.from_config(pickle.load(open(PATH_approx_opt, 'rb'))) # load optimizer config
            self.approximator.fit(self.dataset, epochs=1, steps_per_epoch=steps_per_epoch_3,
                                  callbacks = [model_checkpoint_callback, hist])

            with open('approximator_optimizer_config.pkl', 'wb') as file:
                pickle.dump(self.approximator.optimizer.get_config(), file)  # save optimizer config

            # design
            self.design_loss.joint_model.design_generator.load_state_dict(torch.load(PATH_2)["model_state_dict"])
            optimizer.load_state_dict(torch.load(PATH_2)["optimizer_state_dict"])  
            for _ in tqdm(range(steps_per_epoch_3)):
                with torch.enable_grad():
                    optimizer.zero_grad()
                    history = self.design_loss.simulate_history()
                    taus.append(int(history["n_obs"].item()**2))
                    loss = self.design_loss(history)
                    loss.backward()
                    losses_design.append(loss.item())
                    optimizer.step()

            print(f"Loss: {round(loss.item(), 4)}")
            
            torch.save({'model_state_dict': self.design_loss.joint_model.design_generator.state_dict()}, PATH_2)
            
        with open(os.path.join(PATH, 'bf_loss.pkl'), 'wb') as f:
            pickle.dump(hist.history, f)
        with open(os.path.join(PATH, 'design_loss.pkl'), 'wb') as f:
            pickle.dump(losses_design, f)
        with open(os.path.join(PATH, 'tau.pkl'), 'wb') as f:
            pickle.dump(taus, f)
            

class InferenceDesignApproximatorDesignFirst:
    def __init__(self, 
                 approximator: bf.Approximator, 
                 design_loss: nn.Module, 
                 dataset: keras.utils.PyDataset) -> None:
        
        self.approximator = approximator
        self.design_loss = design_loss
        self.dataset = dataset

    def train(self, PATH: str, **hyper_params):
        PATH = os.path.join("DAD", PATH)
        
        if not os.path.exists(PATH):
            os.makedirs(PATH)

        # Stage 1: Train design network using prior predictive samples (DAD)
        epochs_1 = hyper_params["epochs_1"]
        steps_per_epoch_1 = hyper_params["steps_per_epoch_1"]
        PATH_1 = os.path.join(PATH, "design_network.pt")

        losses_design = []
        taus = []

        trainable_params = [param for param in self.design_loss.joint_model.design_generator.parameters() if param.requires_grad]
        optimizer = Adam(trainable_params, lr=1e-3)

        # use prior predictive
        self.design_loss.set_use_prior_predictive(True)

        print("Stage 1: Train design network using prior predictive samples")
        with torch.enable_grad():
            for e in range(epochs_1):
                print(f"Epoch {e + 1} / {epochs_1}")
                for _ in tqdm(range(steps_per_epoch_1)):
                    optimizer.zero_grad()
                    history = self.design_loss.simulate_history()
                    taus.append(int(history["n_obs"].item()**2))
                    loss = self.design_loss(history)
                    loss.backward()
                    optimizer.step()
                    losses_design.append(loss.item())

                print(f"Loss: {round(loss.item(), 4)}")

                torch.save({'model_state_dict': self.design_loss.joint_model.design_generator.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()}, PATH_1)
                
        torch.save({'model_state_dict': self.design_loss.joint_model.design_generator.state_dict()}, os.path.join(PATH, "design_network_stage_1.pt"))
        
        self.dataset.set_stage(1)
        self.approximator.compile(optimizer="AdamW")

        PATH_2 = os.path.join(PATH, "approximator.weights.h5")

        epochs_2 = hyper_params["epochs_2"]
        steps_per_epoch_2 = hyper_params["steps_per_epoch_2"]

        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            PATH_2,
            verbose=0,
            save_best_only=False,
            save_weights_only=True
        )
        hist = History()

        self.approximator.fit(self.dataset, epochs = epochs_2, steps_per_epoch = steps_per_epoch_2, 
                              callbacks=[model_checkpoint_callback, hist])
        
        self.approximator.save_weights(os.path.join(PATH, "approximator_stage_2.weights.h5"))
        PATH_approx_opt = os.path.join(PATH, 'approximator_optimizer_config.pkl')
        with open(PATH_approx_opt, 'wb') as file:
            pickle.dump(self.approximator.optimizer.get_config(), file)


        # Stage 3: Joint training
        self.dataset.set_stage(3)
        self.design_loss.set_use_prior_predictive(False) # use bf posterior 

        ## load weights and optimizer state from previous stages
        self.design_loss.joint_model.design_generator.load_state_dict(torch.load(PATH_1)["model_state_dict"])
        optimizer.load_state_dict(torch.load(PATH_1)["optimizer_state_dict"]) 
        self.approximator.load_weights(PATH_2)
        self.approximator.optimizer.from_config(pickle.load(open(PATH_approx_opt, 'rb'))) 
        
        epochs_3 = hyper_params["epochs_3"]
        steps_per_epoch_3 = hyper_params["steps_per_epoch_3"]

        print("Stage 3: Joint training")
        for e in range(epochs_3):
            print(f"Epoch {e + 1} / {epochs_3}")

            # bf
            self.approximator.load_weights(PATH_2) # load approximator weights
            self.approximator.optimizer.from_config(pickle.load(open(PATH_approx_opt, 'rb'))) # load optimizer config
            self.approximator.fit(self.dataset, epochs=1, steps_per_epoch=steps_per_epoch_3,
                                  callbacks = [model_checkpoint_callback, hist])

            with open('approximator_optimizer_config.pkl', 'wb') as file:
                pickle.dump(self.approximator.optimizer.get_config(), file)  # save optimizer config

            # design
            self.design_loss.joint_model.design_generator.load_state_dict(torch.load(PATH_1)["model_state_dict"])
            optimizer.load_state_dict(torch.load(PATH_1)["optimizer_state_dict"])  
            for _ in tqdm(range(steps_per_epoch_3)):
                with torch.enable_grad():
                    optimizer.zero_grad()
                    history = self.design_loss.simulate_history()
                    taus.append(int(history["n_obs"].item()**2))
                    loss = self.design_loss(history)
                    loss.backward()
                    losses_design.append(loss.item())
                    optimizer.step()

            print(f"Loss: {round(loss.item(), 4)}")
            
            torch.save({'model_state_dict': self.design_loss.joint_model.design_generator.state_dict()}, PATH_2)
            
        with open(os.path.join(PATH, 'bf_loss.pkl'), 'wb') as f:
            pickle.dump(hist.history, f)
        with open(os.path.join(PATH, 'design_loss.pkl'), 'wb') as f:
            pickle.dump(losses_design, f)
        with open(os.path.join(PATH, 'tau.pkl'), 'wb') as f:
            pickle.dump(taus, f)


class DADOnly:
    def __init__(self, 
                 design_loss: nn.Module) -> None:
        self.design_loss = design_loss

    def train(self, PATH: str, **hyper_params):
        PATH = os.path.join("DAD", PATH)
        
        if not os.path.exists(PATH):
            os.makedirs(PATH)

        epochs_1 = hyper_params["epochs_1"]
        steps_per_epoch_1 = hyper_params["steps_per_epoch_1"]
        PATH_1 = os.path.join(PATH, "design_network.pt")

        losses_design = []
        taus = []

        self.design_loss.set_use_prior_predictive(True)
        
        trainable_params = [param for param in self.design_loss.joint_model.design_generator.parameters() if param.requires_grad]
        optimizer = Adam(trainable_params, lr=1e-3)

        with torch.enable_grad():
            for e in range(epochs_1):
                print(f"Epoch {e + 1} / {epochs_1}")
                for _ in tqdm(range(steps_per_epoch_1)):
                    optimizer.zero_grad()
                    history = self.design_loss.simulate_history()
                    taus.append(int(history["n_obs"].item()**2))
                    loss = self.design_loss(history)
                    loss.backward()
                    optimizer.step()
                    losses_design.append(loss.item())

                print(f"Loss: {round(loss.item(), 4)}")

                torch.save({'model_state_dict': self.design_loss.joint_model.design_generator.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()}, PATH_1)
                
        torch.save({'model_state_dict': self.design_loss.joint_model.design_generator.state_dict()}, 
                    os.path.join(PATH, "design_network_stage_1.pt"))

        with open(os.path.join(PATH, 'design_loss.pkl'), 'wb') as f:
            pickle.dump(losses_design, f)
        with open(os.path.join(PATH, 'tau.pkl'), 'wb') as f:
            pickle.dump(taus, f)
    





