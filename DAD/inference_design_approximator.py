import bayesflow as bf
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
import keras, torch, os

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

        self.approximator.fit(self.dataset, epochs = epochs_1, steps_per_epoch = steps_per_epoch_1)

        path_str = "approximator_stage_1_epoch_" + str(epochs_1) + ".pt"
        PATH_1 = os.path.join(PATH, path_str)

        torch.save({
            'model_state_dict': self.approximator.state_dict(),
            }, 
            PATH_1)

        # Stage 2: Fix BayesFlow, train design network
        self.approximator.summary_network.trainable = False  # Freeze weight for summary network

        epochs_2 = hyper_params["epochs_2"]
        steps_per_epoch_2 = hyper_params["steps_per_epoch_2"]

        trainable_params = [param for param in self.design_loss.joint_model.design_generator.parameters() if param.requires_grad]

        optimizer = Adam(trainable_params, lr=5e-2)
        
        print("Stage 2: Fix BayesFlow weights, train design network")
        for e in range(epochs_2):
            print(f"Epoch {e}")
            for _ in tqdm(range(steps_per_epoch_2)):
                optimizer.zero_grad()
                loss = self.design_loss()
                loss.requires_grad = True
                loss.backward()
                optimizer.step()

            path_str = "design_network_stage_2_epoch_" + str(e) + ".pt"
            PATH_2 = os.path.join(PATH, path_str)
            torch.save({
                'epoch': e,
                'loss': loss,
                'model_state_dict': self.design_loss.joint_model.design_generator.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, 
                PATH_2)
            print(f"Loss: {loss}")

        # Stage 3: Joint training
        self.approximator.summary_network.trainable = True  # Unfreeze weight for summary network
        self.dataset.set_stage(3)

        optimizer = Adam(trainable_params, lr=5e-2)

        epochs_3 = hyper_params["epochs_3"]
        steps_per_epoch_3 = hyper_params["steps_per_epoch_3"]

        print("Stage 3: Joint training")
        for e in range(epochs_3):
            print(f"Epoch {e}")
            # bf
            self.approximator.fit(self.dataset, epochs=1, steps_per_epoch=steps_per_epoch_3)

            for _ in tqdm(range(steps_per_epoch_3)):
                # design network
                optimizer.zero_grad()
                loss = self.design_loss()
                loss.requires_grad = True
                loss.backward()
                optimizer.step()

            path_str_3_1 = "approximator_stage_3_epoch_" + str(e) + ".pt"
            path_str_3_2 = "design_network_stage_3_epoch_" + str(e) + ".pt"
            PATH_3_1 = os.path.join(PATH, path_str_3_1)
            PATH_3_2 = os.path.join(PATH, path_str_3_2)

            torch.save({
                'epoch': e,
                'loss': loss,
                'model_state_dict': self.approximator.state_dict()
                }, 
                PATH_3_1)
            
            torch.save({
                'epoch': e,
                'loss': loss,
                'model_state_dict': self.design_loss.joint_model.design_generator.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, 
                PATH_3_2)

