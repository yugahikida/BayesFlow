import keras

from bayesflow.types import Shape, Tensor
from bayesflow.utils import find_distribution


class InferenceNetwork(keras.Layer):
    def __init__(self, base_distribution: str = "normal", **kwargs):
        super().__init__(**kwargs)
        self.base_distribution = find_distribution(base_distribution)

    def build(self, xz_shape: Shape, conditions_shape: Shape = None) -> None:
        self.base_distribution.build(xz_shape)

    def call(self, xz: Tensor, inverse: bool = False, **kwargs) -> Tensor | tuple[Tensor, Tensor]:
        if inverse:
            return self._inverse(xz, **kwargs)
        return self._forward(xz, **kwargs)

    def _forward(self, x: Tensor, **kwargs) -> Tensor | tuple[Tensor, Tensor]:
        raise NotImplementedError

    def _inverse(self, z: Tensor, **kwargs) -> Tensor | tuple[Tensor, Tensor]:
        raise NotImplementedError

    def sample(self, batch_shape: Shape, conditions: Tensor = None, **kwargs) -> Tensor:
        samples = self.base_distribution.sample(batch_shape)
        samples = self(samples, conditions=conditions, inverse=True, density=False, **kwargs)
        return samples

    def log_prob(self, samples: Tensor, conditions: Tensor = None, **kwargs) -> Tensor:
        _, log_density = self(samples, conditions=conditions, inverse=False, density=True, **kwargs)
        return log_density

    def compute_metrics(self, x: Tensor, conditions: Tensor = None, stage: str = "training") -> dict[str, Tensor]:
        if not self.built:
            xz_shape = keras.ops.shape(x)
            conditions_shape = None if conditions is None else keras.ops.shape(conditions)
            self.build(xz_shape, conditions_shape=conditions_shape)

        metrics = {}

        if stage != "training" and any(self.metrics):
            # compute sample-based metrics
            samples = self.sample((keras.ops.shape(x)[0],), conditions=conditions)

            for metric in self.metrics:
                metrics[metric.name] = metric(samples, x)

        return metrics
