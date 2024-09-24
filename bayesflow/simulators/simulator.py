from collections.abc import Callable
import numpy as np

from bayesflow.types import Shape
from bayesflow.utils import tree_concatenate


class Simulator:
    def sample(self, batch_shape: Shape, **kwargs) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def rejection_sample(
        self,
        batch_shape: Shape,
        predicate: Callable[[dict[str, np.ndarray]], np.ndarray],
        *,
        axis: int = 0,
        sample_size: int = None,
        **kwargs,
    ) -> dict[str, np.ndarray]:
        if sample_size is None:
            sample_shape = batch_shape
        else:
            sample_shape = list(batch_shape)
            sample_shape[axis] = sample_size

            sample_shape = tuple(sample_shape)

        result = {}

        while not result or next(iter(result.values())).shape[axis] < batch_shape[axis]:
            # get a batch of samples
            samples = self.sample(sample_shape, **kwargs)

            # get acceptance mask and turn into indices
            accept = predicate(samples)

            if not isinstance(accept, np.ndarray):
                raise RuntimeError("Predicate must return a numpy array.")

            if accept.shape != (sample_shape[axis],):
                raise RuntimeError(
                    f"Predicate return array must have shape {(sample_shape[axis],)}. " f"Received: {accept.shape}."
                )

            if not accept.dtype == "bool":
                # we could cast, but this tends to hide mistakes in the predicate
                raise RuntimeError("Predicate must return a boolean type array.")

            (accept,) = np.nonzero(accept)

            if not np.any(accept):
                # no samples accepted, skip
                continue

            # apply acceptance mask
            samples = {key: np.take(value, accept, axis=axis) for key, value in samples.items()}

            # concatenate with previous samples
            if not result:
                result = samples
            else:
                result = tree_concatenate([result, samples], axis=axis, numpy=True)

        return result
