import bayesflow as bf
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
import keras
import torch


class InferenceDesignApproximator:
    def __init__(self, 
                 approximator: bf.Approximator, 
                 design_loss: nn.Module, 
                 dataset: keras.utils.PyDataset) -> None:
        
        self.approximator = approximator
        self.design_loss = design_loss
        self.dataset = dataset

    def train(self, PATH: str, **hyper_params):
        # Stage 1: Train Bayesflow, use random design
        num_steps_1 = hyper_params["num_steps_1"]
        # epochs_1 = hyper_params["epochs_1"]; steps_per_epoch_1 = hyper_params["steps_per_epoch_1"]
        self.dataset.set_stage(1)
        self.approximator.compile(optimizer="AdamW")
        epochs_1 = int(num_steps_1 / 100)
        self.approximator.fit(self.dataset, epochs=1, steps_per_epoch = 10)
        path_1 = PATH + "approximator_stage_1"

        # torch.save(self.approximator.state_dict(), path_1)

        # Stage 2: Fix BayesFlow, train design network
        self.approximator.summary_network.trainable = False  # Freeze weight for summary network
        num_steps_2 = hyper_params["num_steps_2"]

        trainable_params = [param for param in self.design_loss.joint_model.design_generator.parameters() if param.requires_grad]

        optimizer = Adam(trainable_params, lr=5e-2)
        # path_2 = PATH + "design_network_stage_2"
        for _ in tqdm(range(num_steps_2)):
            optimizer.zero_grad()
            loss = self.design_loss()
            loss.requires_grad = True
            loss.backward()
            optimizer.step()
            print(loss)

            # torch.save(self.design_loss.joint_model.design_generator.state_dict(), path_2)

        # Stage 3: Joint training
        num_steps_3 = hyper_params["num_steps_3"]
        # epochs_3 = hyper_params["epochs_3"]; steps_per_epoch_3 = hyper_params["steps_per_epoch_3"]
        path_3_1 = PATH + "approximator_stage_3"
        path_3_2 = PATH + "design_network_stage_3"
        for _ in tqdm(range(num_steps_3)):
            # bf
            self.dataset.set_stage(3)
            self.approximator.fit(self.dataset, epochs=1, steps_per_epoch=1)
            # torch.save(self.approximator.state_dict(), path_3_1)

            # design network
            optimizer.zero_grad()
            loss = self.design_loss()
            loss.requires_grad = True
            loss.backward()
            optimizer.step()
            
            # torch.save(self.design_loss.joint_model.design_generator.state_dict(), path_3_2)

