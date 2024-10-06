from custom_simulators import GenericSimulator
import torch, keras
from torch import Tensor

class DataSet(keras.utils.PyDataset):
    def __init__(self, 
                 batch_size: int, 
                 joint_model_1: GenericSimulator, 
                 joint_model_2: GenericSimulator,
                 joint_model_3: GenericSimulator,
                 data_adapter = None,
                 stage: int = None) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.stage = stage # stage 1,2,3
        self.joint_model_1 = joint_model_1
        self.joint_model_2 = joint_model_2
        self.joint_model_3 = joint_model_3
        self.data_adapter = data_adapter
    def set_stage(self, stage: int) -> None:
        self.stage = stage

    def __getitem__(self, item:int) -> dict[str, Tensor]:
        if self.stage == 1: # xi obtained from some stochastic process
            batch = self.joint_model_1.sample(self.batch_size)
            if self.data_adapter is not None:
                batch = self.data_adapter.configure(batch)
            return batch

        if self.stage == 2: # xi obtained from design network taking tau = 0 to learnn inital design (actually don't need it here coz we dont use this for dad part)
            batch = self.joint_model_1.sample(self.batch_size)
            if self.data_adapter is not None:
                batch = self.data_adapter.configure(batch)
            return batch
        
        if self.stage == 3: # xi obtained from design network without tau = 0 (For joint training)
            batch = self.joint_model_1.sample(self.batch_size)
            if self.data_adapter is not None:
                batch = self.data_adapter.configure(batch)
            return batch
    
    
    @property
    def num_batches(self):
        # infinite dataset
        return None