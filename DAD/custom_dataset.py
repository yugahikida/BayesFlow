from custom_simulators import MyGenericSimulator
import torch, keras
from torch import Tensor

class MyDataSet(keras.utils.PyDataset):
    def __init__(self, 
                 batch_shape: torch.Size, 
                 joint_model_1: MyGenericSimulator, 
                 joint_model_2: MyGenericSimulator,
                 stage: int = None) -> None:
        super().__init__()

        self.batch_shape = batch_shape
        self.stage = stage # stage 1,2,3
        self.joint_model_1 = joint_model_1
        self.joint_model_2 = joint_model_2

    def set_stage(self, stage: int) -> None:
        self.stage = stage

    def __getitem__(self, item:int) -> dict[str, Tensor]:
        if self.stage == 1: # xi obtained from some stochastic process
            data = self.joint_model_1.sample(self.batch_shape)
            return data

        if self.stage == 2 or self.stage == 3: # xi obtained from design network
            data = self.joint_model_2.sample(self.batch_shape) 
            return data
    
    @property
    def num_batches(self):
        # infinite dataset
        return None