import numpy as np
from tqdm.notebook import tqdm

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from bayesflow.configuration import *
from bayesflow.exceptions import SimulationError
from bayesflow.helpers import apply_gradients
from bayesflow.amortized_inference import AmortizedPosterior, AmortizedLikelihood, JointAmortizer


class Trainer:

    def __init__(self, amortizer, generative_model=None, configurator=None, optimizer=None,
                 learning_rate=0.0005, checkpoint_path=None, max_to_keep=5, clip_method='global_norm', 
                 clip_value=None, skip_checks=False):
        """Base class for a trainer performing forward inference and training an amortized neural estimator.

        Parameters
        ----------
        amortizer        : bayesflow.amortizers.Amortizer
            The neural architecture to be optimized
        generative_model : bayesflow.forward_inference.GenerativeModel
            A generative model returning a dictionary with randomly sampled parameters, data, and optional context
        configurator     : callable 
            A callable object transforming and combining the outputs of the generative model into inputs for BayesFlow
        optimizer        : tf.keras.optimizer.Optimizer or None
            Optimizer for the neural network. ``None`` will result in `tf.keras.optimizers.Adam`
        learning_rate    : float or tf.keras.schedules.LearningRateSchedule
            The learning rate used for the optimizer
        checkpoint_path  : string, optional
            Optional folder name for storing the trained network
        max_to_keep      : int, optional
            Number of checkpoints to keep
        clip_method      : {'norm', 'value', 'global_norm'}
            Optional gradient clipping method
        clip_value       : float
            The value used for gradient clipping when clip_method is in {'value', 'norm'}
        skip_checks      : boolean
            If True, do not perform consistency checks, i.e., simulator runs and passed through nets
        """

        self.amortizer = amortizer
        self.generative_model = generative_model
        if self.generative_model is None:
            print("Trainer initialization: No generative model provided. Only offline learning mode is available!")
        self.configurator = self._manage_configurator(configurator)

        self.clip_method = clip_method
        self.clip_value = clip_value
        self.n_obs = None

        # Optimizer settings
        if optimizer is None:
            self.optimizer = Adam(learning_rate)
        else:
            self.optimizer = optimizer(learning_rate)

        # Checkpoint settings
        if checkpoint_path is not None:
            self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.network)
            self.manager = tf.train.CheckpointManager(self.checkpoint, checkpoint_path, max_to_keep=max_to_keep)
            self.checkpoint.restore(self.manager.latest_checkpoint)
            if self.manager.latest_checkpoint:
                print("Networks loaded from {}".format(self.manager.latest_checkpoint))
            else:
                print("Initializing networks from scratch.")
        else:
            self.checkpoint = None
            self.manager = None
        self.checkpoint_path = checkpoint_path

        if not skip_checks:
            self._check_consistency()

    def load_pretrained_network(self):
        """Attempts to load a pre-trained network if checkpoint path is provided and a checkpoint manager exists.
        """

        if self.manager is None or self.checkpoint is None:
            return False
        status = self.checkpoint.restore(self.manager.latest_checkpoint)
        return status

    def train_online(self, epochs, iterations_per_epoch, batch_size, **kwargs):
        """Trains the inference network(s) via online learning. Additional keyword arguments
        are passed to the generative mode, configurator, and amortizer.

        Parameters
        ----------
        epochs               : int 
            Number of epochs (and number of times a checkpoint is stored)
        iterations_per_epoch : int 
            Number of batch simulations to perform per epoch
        batch_size           : int 
            Number of simulations to perform at each backprop step

        Returns
        -------
        losses : dict(ep_num : list(losses))
            A dictionary storing the losses across epochs and iterations
        """

        losses = dict()
        for ep in range(1, epochs + 1):
            losses[ep] = []
            with tqdm(total=iterations_per_epoch, desc='Training epoch {}'.format(ep)) as p_bar:
                for it in range(1, iterations_per_epoch + 1):
                    
                    # Obtain generative model outputs
                    forward_dict = self._forward_inference(batch_size, **kwargs)

                    # Configure generative model outputs for amortizer
                    input_dict = self.configurator(forward_dict, **kwargs)

                    # Forward pass and one step backprop
                    loss = self._train_step(input_dict, **kwargs)

                    # Store loss into dictionary
                    losses[ep].append(loss)

                    # Update progress bar
                    p_bar.set_postfix_str("Epoch {0},Iteration {1},Loss: {2:.3f},Running Loss: {3:.3f}"
                                          .format(ep, it, loss, np.mean(losses[ep])))
                    p_bar.update(1)

            # Store after each epoch, if specified
            if self.manager is not None:
                self.manager.save()
        return losses

    def _forward_inference(self, n_sim, *args):
        """
        Performs one step of single-model forward inference.

        Parameters
        ----------
        n_sim : int
            Number of simulations to perform at the given step (i.e., batch size)

        Returns
        -------
        out_dict : dict
            The outputs of the generative model.

        Raises
        ------
        SimulationError
            If the trainer has no generative model but `trainer._forward_inference`
            is called (i.e., needs to simulate data from the generative model)
        """

        if self.generative_model is None:
            raise SimulationError("No generative model specified. Only offline learning is available!")
        out_dict = self.generative_model(n_sim, *args)
        return out_dict

    def _train_step(self, input_dict, **kwargs):
        """Computes loss and applies gradients.
        """

        # Forward pass and loss computation
        with tf.GradientTape() as tape:
            loss = self.amortizer.compute_loss(input_dict, **kwargs)

        # One step backprop
        gradients = tape.gradient(loss, self.amortizer.trainable_variables)
        apply_gradients(self.optimizer, gradients, self.amortizer.trainable_variables, 
                        self.clip_value, self.clip_method)

        return loss.numpy()

    def _manage_configurator(self, config_fun):
        """ Determines which configurator to use if None specified.        
        """

        # Do nothing if callable provided
        if callable(config_fun):
            return config_fun

        # If None (default), infer default config based on amortizer type
        else:
            # Amortized posterior
            if type(self.amortizer) is AmortizedPosterior:
                default_config = DefaultPosteriorConfigurator
                default_combiner = DefaultPosteriorCombiner()
                default_transfomer = DefaultPosteriorTransformer()

            # Amortized lieklihood
            elif type(self.amortizer) is AmortizedLikelihood:
                default_config = DefaultLikelihoodConfigurator
                default_combiner = DefaultLikelihoodTransformer()
                default_transfomer = DefaultLikelihoodCombiner()

            # Joint amortizer
            elif type(self.amortizer) is JointAmortizer:
                default_config = DefaultJointConfigurator
                default_combiner = DefaultJointTransformer()
                default_transfomer = DefaultJointCombiner()
            
            # Unknown raises an error
            else:
                raise NotImplementedError(f"Could not initialize configurator based on" +
                                           "amortizer type {type(self.amortizer)}!")

        # Check string types
        if type(config_fun) is str:
            if config_fun == "variable_num_obs":
                return default_config(
                    transform_fun=VariableObservationsTransformer(),
                    combine_fun=default_combiner)

            elif config_fun == 'one_hot':
                return default_config(
                    transform_fun=OneHotTransformer(),
                    combine_fun=default_combiner)

            elif config_fun == 'variable_num_obs_one_hot':
                return default_config(
                    transform_fun=TransformerUnion([
                        VariableObservationsTransformer(),
                        OneHotTransformer(),
                    ]),
                    combine_fun=default_combiner)
            elif config_fun == 'one_hot_variable_num_obs':
                return default_config(
                    transform_fun=TransformerUnion([
                        OneHotTransformer(),
                        VariableObservationsTransformer(),
                    ]),
                    combine_fun=default_combiner)
            else:
                raise NotImplementedError(f"Could not initialize configurator based on string" +
                                           "argument should be in {STRING_CONFIGS}")

        elif config_fun is None:
            config_fun = default_config(
                transform_fun=default_transfomer,
                combine_fun=default_combiner
            )
            return config_fun
        
        else:
            raise NotImplementedError(f"Could not infer configurator based on provided type: {type(config_fun)}")
    
    def _check_consistency(self):
        """Attempts to run one step generative_model -> configurator -> amortizer -> loss."""

        if self.generative_model is not None:
            _n_sim = 2
            try: 
                print('Performing a consistency check with provided modules...', end="")
                _ = self.amortizer.compute_loss(self.configurator(self.generative_model(_n_sim)))
                print('Done.')
            except Exception as err:
                print(err)
                raise ConfigurationError("Could not carry out computations of generative_model -> configurator -> amortizer -> loss!")