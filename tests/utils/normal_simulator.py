import numpy as np

from bayesflow.simulators import Simulator
from bayesflow.types import Shape, Tensor


class NormalSimulator(Simulator):
    """TODO: Docstring"""

    def sample(self, batch_shape: Shape, num_observations: int = 32) -> dict[str, Tensor]:
        mean = np.random.normal(0.0, 0.1, size=batch_shape + (2,))
        mean = np.repeat(mean[:, None], num_observations, axis=1)

        std = np.random.lognormal(0.0, 0.1, size=batch_shape + (2,))
        std = np.repeat(std[:, None], num_observations, axis=1)

        noise = np.random.standard_normal(batch_shape + (num_observations, 2))

        x = mean + std * noise
        return dict(mean=mean, std=std, x=x)
