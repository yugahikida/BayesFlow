import tensorflow as tf
import numpy as np
import bayesflow as bf
from bayesflow.amortizers import AmortizedPosterior
from bayesflow.default_settings import DEFAULT_KEYS
from collections import namedtuple

SurrogateOutput = namedtuple("SurrogateOutput", ["z", "x1", "nll", "surrogate"])

def compute_volume_change(jac):
    full_dimensional = jac.shape[-1] == jac.shape[-2]
    if full_dimensional:
        return tf.linalg.slogdet(jac)[1]
    else:
        if jac.shape[-1] < jac.shape[-2]:
            jac = tf.transpose(jac, perm=[0, 2, 1])
        jac_transpose_jac = tf.linalg.matmul(jac, tf.transpose(jac, perm=[0, 2, 1]))
        return tf.linalg.slogdet(jac_transpose_jac)[1] / 2

class FFFAmortizedPosterior(AmortizedPosterior):
    def __init__(
        self,
        beta: int = 1.0,
        surrogate: bool = True,
        memorize_last_samples: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.beta = beta
        self.surrogate = surrogate
        if memorize_last_samples:
            self.last_batch = {
                "inputs": None,
                "reconstruction": None,
                "latent": None,
            }

    def _sample_v(self, x: tf.Tensor, hutchinson_samples: int) -> tf.Tensor:
        if hutchinson_samples > x.shape[-1]:
            raise ValueError(
                f"Too many Hutchinson samples: got {hutchinson_samples}, expected <= {x.shape[-1]}"
            )
        if len(x.shape) != 2:
            raise ValueError(f"Input must be a vector, got shape {x.shape}")
        v = tf.random.normal(x.shape + (hutchinson_samples,))
        q, r = tf.linalg.qr(v)
        return q * tf.sqrt(float(x.shape[-1]))

    def _log_prob(
        self,
        input_dict,
        validation,
        use_surrogate: bool = True,
        hutchinson_samples: int = 1,
        return_summary: bool = False,
        **kwargs,
    ) -> tf.Tensor:
        summary_out, full_cond = self._compute_summary_condition(
            input_dict.get(DEFAULT_KEYS["summary_conditions"]),
            input_dict.get(DEFAULT_KEYS["direct_conditions"]),
            **kwargs,
        )

        log_det_jac = None
        if use_surrogate and not validation:
            surrogate = 0
            with tf.GradientTape() as tape_vjp:
                x = tf.convert_to_tensor(input_dict[DEFAULT_KEYS["parameters"]])
                tape_vjp.watch(x)
                z = self.inference_net(x, full_cond, **kwargs)
                if isinstance(z, tuple):
                    z, _ = z

                vs = self._sample_v(z, hutchinson_samples)

                for k in range(hutchinson_samples):
                    v = vs[..., k]

                    # calculate g'(z) v via forward-mode AD
                    accumulator = tf.autodiff.ForwardAccumulator(z, v)
                    with accumulator:
                        x1 = self.inference_net(z, full_cond, inverse=True, **kwargs)
                    jvp = accumulator.jvp(x1)

                    # calculate v  f'(z) via backward-mode AD
                    vjp = tape_vjp.gradient(z, x, output_gradients=v)
                    surrogate += (
                        tf.math.reduce_sum(vjp * tf.stop_gradient(jvp), axis=-1)
                        / hutchinson_samples
                    )

            nll = tf.reduce_mean(- tf.norm(z, axis=-1) - surrogate)
            log_det_jac = surrogate
        else:
            with tf.GradientTape() as tape_enc:
                x = tf.convert_to_tensor(input_dict[DEFAULT_KEYS["parameters"]])
                tape_enc.watch(x)
                z = self.inference_net(x, full_cond, **kwargs)
                if isinstance(z, tuple):
                    z, _ = z
                x1 = self.inference_net(z, full_cond, inverse=True, **kwargs)
                if log_det_jac is None:
                    J = tape_enc.batch_jacobian(z, x)
                    log_det_jac = [compute_volume_change(J[i]) for i in range(tf.shape(J)[0])]
                    log_det_jac = tf.stack(log_det_jac)
            nll = tf.reduce_mean(tf.reduce_sum(tf.square(z), axis=-1) / 2 - log_det_jac)

        return SurrogateOutput(z, x1, nll, log_det_jac), summary_out, full_cond

    def _reconstruction_loss(self, a, b):
        reconstruction_loss = tf.reduce_sum(
            tf.square(a - b), axis=tuple(range(1, len(a.shape)))
        )
        reconstruction_loss = tf.reduce_mean(reconstruction_loss)
        return reconstruction_loss

    def compute_loss(self, input_dict, validation=False, **kwargs):
        # calculate surrogate
        log_prob_output, sum_out, full_cond = self._log_prob(
            input_dict, validation, use_surrogate=self.surrogate, **kwargs
        )
        z, x1, nll_gauss, J = log_prob_output

        # optional loss terms in bayesflow
        # Case summary loss should be computed
        if self.summary_loss is not None:
            sum_loss = self.summary_loss(sum_out)
        # Case no summary loss, simply add 0 for convenience
        else:
            sum_loss = 0.0

        # Case dynamic latent space - function of summary conditions
        if self.latent_is_dynamic:
            logpdf = self.latent_dist(sum_out).log_prob(z)
        # Case _static latent space
        else:
            logpdf = self.latent_dist.log_prob(z)

        x = self.inference_net(
            z, full_cond, inverse=True, **kwargs
        )  # TODO: use x from surrogate_output for efficiency
        reconstruction_loss = self._reconstruction_loss(
            input_dict[DEFAULT_KEYS["parameters"]], x
        )

        # Compute and return total posterior and reconstruction loss
        nll = tf.reduce_mean(-logpdf - J)
        #total_loss = nll + self.beta * reconstruction_loss + sum_loss

        if hasattr(self, "last_batch"):
            self.last_batch["inputs"] = input_dict
            self.last_batch["reconstruction"] = x
            self.last_batch["latent"] = z

        # return nll
        return {
            "nll": nll,
            "reconstruction_loss": self.beta * reconstruction_loss,
        }

    def sample(self, input_dict, n_samples, to_numpy=True, **kwargs):
        """Generates random draws from the approximate posterior given a dictionary with conditonal variables.

        Parameters
        ----------
        input_dict  : dict
            Input dictionary containing at least one of the following mandatory keys, if ``DEFAULT_KEYS`` unchanged:
            ``summary_conditions`` : the conditioning variables (including data) that are first passed through a summary network
            ``direct_conditions``  : the conditioning variables that the directly passed to the inference network
        n_samples   : int
            The number of posterior draws (samples) to obtain from the approximate posterior
        to_numpy    : bool, optional, default: True
            Flag indicating whether to return the samples as a ``np.ndarray`` or a ``tf.Tensor``.
        **kwargs    : dict, optional, default: {}
            Additional keyword arguments passed to the networks

        Returns
        -------
        post_samples : tf.Tensor or np.ndarray of shape (n_data_sets, n_samples, n_params)
            The sampled parameters from the approximate posterior of each data set
        """

        # Compute learnable summaries, if appropriate
        _, conditions = self._compute_summary_condition(
            input_dict.get(DEFAULT_KEYS["summary_conditions"]),
            input_dict.get(DEFAULT_KEYS["direct_conditions"]),
            training=False,
            **kwargs,
        )

        # Obtain random draws from the approximate posterior given conditioning variables
        # Case dynamic, assume tensorflow_probability instance, so need to reshape output from
        # (n_samples, n_data_sets, latent_dim) to (n_data_sets, n_samples, latent_dim)
        if self.latent_is_dynamic:
            z_samples = self.latent_dist(conditions).sample(n_samples)
            z_samples = tf.transpose(z_samples, (1, 0, 2))
        # Case _static latent - marginal samples from the specified dist
        else:
            z_samples = self.latent_dist.sample(n_samples)

        # Obtain random draws from the approximate posterior given conditioning variables
        cond = np.tile(conditions, n_samples).reshape(n_samples, *conditions.shape)
        post_samples = self.inference_net(
            z_samples, cond, training=False, inverse=True, **kwargs
        )

        # Only return 2D array, if first dimensions is 1
        if post_samples.shape[0] == 1:
            post_samples = post_samples[0]
        self._check_output_sanity(post_samples)

        # Return numpy version of tensor or tensor itself
        if to_numpy:
            return post_samples.numpy()
        return post_samples

    def sample_loop(self, input_list, n_samples, to_numpy=True, **kwargs):
        """Generates random draws from the approximate posterior given a list of dicts with conditonal variables.
        Useful when GPU memory is limited or data sets have a different (non-Tensor) structure.

        Parameters
        ----------
        input_list  : list of dictionaries, each dictionary having the following mandatory keys, if ``DEFAULT_KEYS`` unchanged:
            ``summary_conditions`` : the conditioning variables (including data) that are first passed through a summary network
            ``direct_conditions``  : the conditioning variables that the directly passed to the inference network
        n_samples   : int
            The number of posterior draws (samples) to obtain from the approximate posterior
        to_numpy    : bool, optional, default: True
            Flag indicating whether to return the samples as a ``np.ndarray`` or a ``tf.Tensor``
        **kwargs    : dict, optional, default: {}
            Additional keyword arguments passed to the networks

        Returns
        -------
        post_samples : tf.Tensor or np.ndarray of shape (n_datasets, n_samples, n_params)
            The sampled parameters from the approximate posterior of each data set
        """

        post_samples = []
        for input_dict in input_list:
            post_samples.append(self.sample(input_dict, n_samples, to_numpy, **kwargs))
        if to_numpy:
            return np.concatenate(post_samples, axis=0).reshape(
                len(input_list), n_samples, -1
            )
        return tf.concat(post_samples, axis=0).reshape(len(input_list), n_samples, -1)

    def get_last_batch(self):
        if hasattr(self, "last_batch"):
            return self.last_batch
