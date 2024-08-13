import bayesflow as bf
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
import keras
import torch
import os
import pickle

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

        self.approximator.fit(self.dataset, epochs = epochs_1, steps_per_epoch = steps_per_epoch_1, 
                              callbacks=[model_checkpoint_callback])
        
        self.approximator.save_weights(os.path.join(PATH, "approximator_stage_1.weights.h5"))
        PATH_approx_opt = os.path.join(PATH, 'approximator_optimizer_config.pkl')
        with open(PATH_approx_opt, 'wb') as file:
            pickle.dump(self.approximator.optimizer.get_config(), file)

        # Stage 2: Fix BayesFlow, train design network
        self.approximator.summary_network.trainable = False  # Freeze weight for summary network

        epochs_2 = hyper_params["epochs_2"]
        steps_per_epoch_2 = hyper_params["steps_per_epoch_2"]

        trainable_params = [param for param in self.design_loss.joint_model.design_generator.parameters() if param.requires_grad]
        optimizer = Adam(trainable_params, lr=5e-2)
        PATH_2 = os.path.join(PATH, "design_network.pt")
        
        print("Stage 2: Fix BayesFlow weights, train design network")

        with torch.enable_grad():
            for e in range(epochs_2):
                print(f"Epoch {e + 1} / {epochs_2}")
                for _ in tqdm(range(steps_per_epoch_2)):
                    optimizer.zero_grad()
                    loss = self.design_loss()
                    loss.backward()
                    optimizer.step()

                print(f"Loss: {round(loss.item(), 4)}")

                torch.save({'epoch': e,
                            'loss': loss,
                            'model_state_dict': self.design_loss.joint_model.design_generator.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()}, 
                            PATH_2)
                
        torch.save({'epoch': e,
                    'loss': loss,
                    'model_state_dict': self.design_loss.joint_model.design_generator.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, 
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
        
        # PATH_3_1 = os.path.join(PATH, "approximator_stage_3.weights.h5")
        # PATH_3_2 = os.path.join(PATH, "design_network_stage_3.pt")

        print("Stage 3: Joint training")
        for e in range(epochs_3):
            print(f"Epoch {e + 1} / {epochs_3}")

            # bf
            self.approximator.load_weights(PATH_1) # load approximator weights
            self.approximator.optimizer.from_config(pickle.load(open(PATH_approx_opt, 'rb'))) # load optimizer config
            self.approximator.fit(self.dataset, epochs=1, steps_per_epoch=steps_per_epoch_3,
                                  callbacks = [model_checkpoint_callback])

            with open('approximator_optimizer_config.pkl', 'wb') as file:
                pickle.dump(self.approximator.optimizer.get_config(), file)  # save optimizer config

            # design
            self.design_loss.joint_model.design_generator.load_state_dict(torch.load(PATH_2)["model_state_dict"])
            optimizer.load_state_dict(torch.load(PATH_2)["optimizer_state_dict"])  
            for _ in tqdm(range(steps_per_epoch_3)):
                with torch.enable_grad():
                    optimizer.zero_grad()
                    loss = self.design_loss()
                    loss.backward()
                    optimizer.step()

            print(f"Loss: {round(loss.item(), 4)}")
            
            torch.save({'epoch': e,
                        'loss': loss,
                        'model_state_dict': self.design_loss.joint_model.design_generator.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}, 
                        PATH_2)
            

        # def design(PATH: str, history_o: dict): # PATH is same PATH as used in tran
        #     """
        #     obtain next design given observed history
        #     """
        #     PATH_3_2 = os.path.join(PATH, "design_network_stage_3.pt")

        #     # load weights
        #     self.design_loss.joint_model.design_generator.load_state_dict(torch.load(PATH_3_2)["model_state_dict"])

        #     next_design = self.design_loss.joint_model.design_generator(history_o)

        #     return next_design.detach().numpy()





