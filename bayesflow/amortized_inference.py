# Copyright 2022 The BayesFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf

from bayesflow.exceptions import ConfigurationError, SummaryStatsError
from bayesflow.losses import kl_latent_space_gaussian, kl_latent_space_student, mmd_summary_space
from bayesflow.default_settings import DEFAULT_KEYS


class AmortizedPosterior(tf.keras.Model):
    """ An interface to connect an inference network for parameter estimation with an optional summary network
    as in the original BayesFlow set-up described in the paper:

    Radev, S. T., Mertens, U. K., Voss, A., Ardizzone, L., & Köthe, U. (2020). 
    BayesFlow: Learning complex stochastic models with invertible neural networks. 
    IEEE transactions on neural networks and learning systems.

    But also allowing for augmented functionality, such as model misspecification detection in summary space:

    Schmitt, M., Bürkner, P. C., Köthe, U., & Radev, S. T. (2021). 
    BayesFlow can reliably detect Model Misspecification and Posterior Errors in Amortized Bayesian Inference. 
    arXiv preprint arXiv:2112.08866.

    And learning of fat-tailed posteriors with a Student-t latent pushforward density:

    Jaini, P., Kobyzev, I., Yu, Y., & Brubaker, M. (2020, November). 
    Tails of lipschitz triangular flows. 
    In International Conference on Machine Learning (pp. 4673-4681). PMLR.
    """

    def __init__(self, inference_net, summary_net=None, loss_fun=None, summary_loss_fun=None):
        """Initializes a composite neural network to represent an amortized approximate posterior.

        Parameters
        ----------
        inference_net     : tf.keras.Model
            An (invertible) inference network which processes the outputs of a generative model 
        summary_net       : tf.keras.Model or None, optional, default: None
            An optional summary network
        loss_fun          : callable or None, optional, default: None
            The loss function which accepts the outputs of the amortizer. If None, the loss is inferred
            based on the `inference_net` type. 
        summary_loss_fun  : callable or None, optional, default: None
            The loss function which accepts the outputs of the summary network. If None, no loss is provided.

        Important
        ----------
        - If no `summary_net` is provided, then the output dictionary of your generative model should not contain
        any `sumamry_conditions`, i.e., `summary_conditions` should be set to None, otherwise these will be ignored.

        - If no custom `loss_fun` is provided, the loss function will either be a Kullback-Leibler (KL) divergence
        for a latent Gaussian space or a KL for a latent Student-t space, depending on the existence of `tail_network`
        attribute in the inference net. If you are using a custom inference net mapping parameters to a latent Student-t
        base distribution, make sure the inference net has a `tail_network` attribute.
        """

        super(AmortizedPosterior, self).__init__()

        self.inference_net = inference_net
        self.summary_net = summary_net
        self.summary_loss = self._determine_summary_loss(summary_loss_fun)
        self.inference_loss = self._determine_loss(loss_fun)

    def call(self, input_dict, return_summary=False, **kwargs):
        """ Performs a forward pass through the summary and inference network.

        Parameters
        ----------
        input_dict     : dict  
            Input dictionary containing the following mandatory keys, if DEFAULT keys unchanged: 
            `parameters`         - the latent model parameters over which a condition density is learned
            `summary_conditions` - the conditioning variables (including data) that are first passed through a summary network
            `direct_conditions`  - the conditioning variables that the directly passed to the inference network
        return_summary : bool, optional, default: False
            A flag which determines whether the learnable data summaries (representations) are returned or not.

        Returns
        -------
        net_out or (net_out, summary_out) : tuple of tf.Tensor
            the outputs of ``inference_net(theta, summary_net(x, c_s), c_d)``, usually a latent variable and
            log(det(Jacobian)), that is a tuple ``(z, log_det_J) or (sum_outputs, (z, log_det_J)) if 
            return_summary is set to True and a summary network is defined.`` 
        """
        
        # Concatenate conditions, if given
        summary_out, full_cond = self._compute_summary_condition(
            input_dict.get(DEFAULT_KEYS['summary_conditions']), 
            input_dict.get(DEFAULT_KEYS['direct_conditions']),
            **kwargs
        )

        # Compute output of inference net
        net_out = self.inference_net(input_dict[DEFAULT_KEYS['parameters']], full_cond, **kwargs)

        if not return_summary:
            return net_out
        return net_out, summary_out

    def sample(self, input_dict, n_samples, to_numpy=True, **kwargs):
        """ Generates random draws from the approximate posterior given conditonal variables.

        Parameters
        ----------
        input_dict  : dict  
            Input dictionary containing the following mandatory keys, if DEFAULT KEYS unchanged: 
            `summary_conditions` : the conditioning variables (including data) that are first passed through a summary network
            `direct_conditions`  : the conditioning variables that the directly passed to the inference network
        n_samples   : int
            The number of posterior draws (samples) to obtain from the approximate posterior
        to_numpy    : bool, optional, default: True
            Flag indicating whether to return the samples as a `np.array` or a `tf.Tensor`

        Returns
        -------
        post_samples : tf.Tensor or np.ndarray of shape (n_datasets, n_samples, n_params)
            the sampled parameters per data set
        """

        # Compute learnable summaries, if appropriate
        _, condition = self._compute_summary_condition(
            input_dict.get(DEFAULT_KEYS['summary_conditions']), 
            input_dict.get(DEFAULT_KEYS['direct_conditions']), 
            **kwargs
        )

        # Obtain random draws from the approximate posterior given conditioning variables
        post_samples = self.inference_net.sample(condition, n_samples, **kwargs)

        if to_numpy:
            return post_samples.numpy()
        return post_samples

    def log_posterior(self, input_dict, to_numpy=True, **kwargs):
        """ Calculates the approximate log-posterior of targets given conditional variables via
        the change-of-variable formula for a conditional normalizing flow.

        Parameters
        ----------
        input_dict : dict  
            Input dictionary containing the following mandatory keys, if DEFAULT_KEYS unchanged: 
            `parameters`         : the latent model parameters over which a conditional density (i.e., a posterior) is learned
            `summary_conditions` : the conditioning variables (including data) that are first passed through a summary network
            `direct_conditions`  : the conditioning variables that are directly passed to the inference network
        to_numpy  : bool, optional, default: True
            Flag indicating whether to return the lpdf values as a `np.array` or a `tf.Tensor`

        Returns
        -------
        log_post  : tf.Tensor of shape (batch_size, n_obs)
            the approximate log-posterior density of each each parameter 
        """

        # Compute learnable summaries, if appropriate
        _, conditions = self._compute_summary_condition(
            input_dict.get(DEFAULT_KEYS['summary_conditions']), 
            input_dict.get(DEFAULT_KEYS['direct_conditions']), 
            **kwargs
        )

        # Compute approximate log posterior of provided parameters
        log_post = self.inference_net.log_density(input_dict[DEFAULT_KEYS['parameters']], conditions, **kwargs)

        if to_numpy:
            return log_post.numpy()
        return log_post

    def _compute_summary_condition(self, summary_conditions, direct_conditions, **kwargs):
        """ Determines how to concatenate the provided conditions.
        """

        # Compute learnable summaries, if given
        if self.summary_net is not None:
            sum_condition = self.summary_net(summary_conditions, **kwargs)
        else:
            sum_condition = None

        # Concatenate learnable summaries with fixed summaries, this 
        if sum_condition is not None and direct_conditions is not None:
            full_cond = tf.concat([sum_condition, direct_conditions], axis=-1)
        elif sum_condition is not None:
            full_cond = sum_condition
        elif direct_conditions is not None:
            full_cond = direct_conditions
        else:
            raise SummaryStatsError("Could not concatenarte or determine conditioning inputs...")
        return sum_condition, full_cond

    def _determine_loss(self, loss_fun):
        """ Determines which loss to use if default None argument provided, otherwise return argument.
        """

        if loss_fun is None:
            try:
                if self.inference_net.tail_network is not None:
                    return kl_latent_space_student
                else:
                    return kl_latent_space_gaussian
            except Exception as _:
                raise ConfigurationError("Could not infer loss function based on inference net type. " +
                                         "Please provide a custom loss function!")
        elif callable(loss_fun):
            return loss_fun
        else:
            raise ConfigurationError("Loss function is neither default not callable. Please provide a valid loss function!")

    def _determine_summary_loss(self, loss_fun):
        """ Determines which summary loss to use if default None argument provided, otherwise return argument.
        """

        # If callable, return provided loss
        if loss_fun is None or callable(loss_fun):
            return loss_fun
        
        # If string, check for MMD or mmd
        elif type(loss_fun) is str:
            if loss_fun.lower() == 'mmd':
                return mmd_summary_space
            else:
                raise NotImplementedError("For now, only 'mmd' is supported as a string argument for summary_loss_fun!")
        # Throw if loss type unexpected
        else:
            raise NotImplementedError("Could not infer summary_loss_fun, argument should be of type (None, callable, or str)!")

    def compute_loss(self, input_dict, **kwargs):
        """ Computes the loss of the posterior amortizer given an input dictionary.

        Parameters
        ----------
        input_dict : dict  
            Input dictionary containing the following mandatory keys: 
            `parameters`         - the latent model parameters over which a condition density is learned
            `summary_conditions` - the conditioning variables that are first passed through a summary network
            `direct_conditions`  - the conditioning variables that the directly passed to the inference network

        Returns
        -------
        loss      : tf.Tensor of shape (1,) - the total computed loss given input variables
        """

        if self.summary_loss is not None:
            net_out, sum_out = self(input_dict, return_summary=True, **kwargs)
            loss =  self.inference_loss(*net_out) + self.summary_loss(sum_out)
        else:
            net_out = self(input_dict, **kwargs)
            loss = self.inference_loss(*net_out)
        return loss


class AmortizedLikelihood(tf.keras.Model):
    """ An interface for a surrogate model of a simulator, or an implicit likelihood
    ``p(params | data, context).''
    """

    def __init__(self, surrogate_net, loss_fun=None):
        """Initializes a composite neural architecture representing an amortized emulator 
        for the simulator (i.e., the implicit likelihood model).

        Parameters
        ----------
        surrogate_net : tf.keras.Model
            An (invertible) inference network which processes the outputs of the generative model.
        loss_fun      : callable or None, optional, default: None
            The loss function which accepts the outputs of the amortizer. If None, the loss is inferred
            based on the `surrogate_net` type. 

        Important
        ----------
        - If no custom `loss_fun` is provided, the loss function will either be a Kullback-Leibler (KL) divergence
        for a latent Gaussian space or a KL for a latent student-t space, depending on the existence of `tail_network`
        attribute in the `surrogate_net`. If you are using a custom `surrogate_net` mapping parameters to a latent student-t
        base distribution, make sure the `surrogate_net` has a `tail_network` attribute.

        """

        super(AmortizedLikelihood, self).__init__()

        self.surrogate_net = surrogate_net
        self.loss = self._determine_loss(loss_fun)

    def call(self, input_dict, **kwargs):
        """ Performs a forward pass through the summary and inference network.

        Parameters
        ----------
        input_dict  : dict 
            Input dictionary containing the following mandatory keys: 
            `observables` - the observables over which a condition density is learned (i.e., the data)
            `conditions`  - the conditioning variables that the directly passed to the inference network
            
        Returns
        -------
        net_out
            the outputs of ``inference_net(theta, summary_net(x, c_s), c_d)``, usually a latent variable and
            log(det(Jacobian)), that is a tuple ``(z, log_det_J)`` or (sum_outputs, (z, log_det_J)) if 
            return_summary is set to True and a summary network is defined.`` 
        """

        net_out = self.surrogate_net(
            input_dict[DEFAULT_KEYS['observables']], 
            input_dict[DEFAULT_KEYS['conditions']], 
            **kwargs)
        return net_out

    def sample(self, input_dict, n_samples, to_numpy=True, **kwargs):
        """ Generates `n_samples` random draws from the surrogate likelihood given input conditions.

        Parameters
        ----------

        input_dict   : dict  
            Input dictionary containing the following mandatory keys, if DEFAULT_KEYS unchanged: 
            `conditions` - the conditioning variables that the directly passed to the inference network
        n_samples    : int
            The number of posterior samples to obtain from the approximate posterior
        to_numpy  : bool, optional, default: True
            Flag indicating whether to return the samples as a `np.array` or a `tf.Tensor`

        Returns
        -------
        lik_samples : tf.Tensor or np.ndarray of shape (n_datasets, n_samples, None)
            Simulated batch of observables from the surrogate likelihood.
        """

        # Obtain random draws from the approximate posterior given conditioning variables
        lik_samples = self.surrogate_net.sample(input_dict[DEFAULT_KEYS['conditions']], n_samples, **kwargs)
        if to_numpy:
            return lik_samples.numpy()
        return lik_samples

    def log_likelihood(self, input_dict, to_numpy=True, **kwargs):
        """ Calculates the approximate log-likelihood of targets given conditional variables via
        the change-of-variable formula for a conditional normalizing flow.

        Parameters
        ----------
        input_dict : dict  
            Input dictionary containing the following mandatory keys, if DEFAULT_KEYS unchanged: 
            `observables` - the variables over which a condition density is learned (i.e., the observables)
            `conditions`  - the conditioning variables that the directly passed to the inference network
        to_numpy   : bool, optional, default: True
            Boolean flag indicating whether to return the log-lik values as a `np.array` or a `tf.Tensor`

        Returns
        -------
        log_lik    : tf.Tensor of shape (batch_size, n_obs)
            the approximate log-likelihood of each data point in each data set
        """

        log_lik = self.surrogate_net.log_density(
            input_dict[DEFAULT_KEYS['observables']], 
            input_dict[DEFAULT_KEYS['conditions']], **kwargs)

        if to_numpy:
            return log_lik.numpy()
        return log_lik

    def _determine_loss(self, loss_fun):
        """ Determines which loss to use if None given, otherwise return provided argument.
        """

        if loss_fun is None:
            try:
                if self.surrogate_net.tail_network is not None:
                    return kl_latent_space_student
                else:
                    return kl_latent_space_gaussian
            except Exception as _:
                raise ConfigurationError("Could not infer loss function based on surrogate_net type. Please input a loss function!")
        elif callable(loss_fun):
            return loss_fun
        else:
            raise ConfigurationError("Loss function is neither default not callable. Please provide a valid loss function!")

    def compute_loss(self, input_dict, **kwargs):
        """ Computes the loss of the amortized given input data provided in input_dict.

        Parameters
        ----------
        input_dict  : dict 
            Input dictionary containing the following mandatory keys: 
            `data`        - the observables over which a condition density is learned (i.e., the observables)
            `conditions`  - the conditioning variables that the directly passed to the inference network

        Returns
        -------
        loss        : tf.Tensor of shape (1,) - the total computed loss given input variables
        """

        net_out = self(input_dict, **kwargs)
        loss =  self.loss(*net_out)
        return loss
            

class JointAmortizer(tf.keras.Model):
    """ An interface for jointly learning a surrogate model of the simulator and an approximate
    posterior given a generative model.
    """

    def __init__(self, amortized_posterior, amortized_likelihood):
        """Initializes a joint learner comprising an amortized posterior and an amortized emulator.

        Parameters
        ----------
        amortized_posterior  : an instance of AmortizedPosterior or tf.keras.Model
            The generative neural posterior approximator.
        amortized_likelihood : an instance of AmortizedEmulator or tf.keras.Model
            The generative neural likelihood approximator.

        Important
        ---------
        #TODO
        """

        super(JointAmortizer, self).__init__()

        self.amortized_posterior = amortized_posterior
        self.amortized_likelihood = amortized_likelihood

    def call(self, input_dict, **kwargs):
        """ Performs a forward pass through both amortizers.

        Parameters
        ----------
        input_dict  : dict 
            Input dictionary containing the following mandatory keys: 
            `posterior_inputs`  - The input dictionary for the amortized posterior
            `likelihood_inputs` - The input dictionary for the amortized likelihood

        Returns
        -------
        (post_out, lik_out) : tuple
            The outputs of the posterior and likelihood networks given input variables.
        """

        post_out = self.amortized_posterior(input_dict['posterior_inputs'], **kwargs)
        lik_out = self.amortized_likelihood(input_dict['likelihood_inputs'], **kwargs)
        return post_out, lik_out

    def compute_loss(self, input_dict, **kwargs):
        """ Computes the loss of the join amortizer by summing the corresponding amortized posterior 
        and likelihood losses.

        Parameters
        ----------
        input_dict  : dict 
            Nested input dictionary containing the following mandatory keys, if DEFAULT_KEYS unchanged:: 
            `posterior_inputs`  - The input dictionary for the amortized posterior
            `likelihood_inputs` - The input dictionary for the amortized likelihood

        Returns
        -------
        total_loss  : tf.Tensor of shape (1,) - the total computed loss given input variables
        """

        loss_post = self.amortized_posterior.compute_loss(input_dict[DEFAULT_KEYS['posterior_inputs']], **kwargs)
        loss_lik = self.amortized_likelihood.compute_loss(input_dict[DEFAULT_KEYS['likelihood_inputs']], **kwargs)
        total_loss = loss_post + loss_lik
        return total_loss

    def log_likelihood(self, input_dict, to_numpy=True, **kwargs):
        """ Calculates the approximate log-likelihood of data given conditional variables via
        the change-of-variable formula for conditional normalizing flows.

        Parameters
        ----------
        input_dict : dict  
            Input dictionary containing the following mandatory keys, if DEFAULT_KEYS unchanged:
            `observables` - the variables over which a condition density is learned (i.e., the observables)
            `conditions`  - the conditioning variables that are directly passed to the inference network
        to_numpy   : bool, optional, default: True
            Flag indicating whether to return the samples as a `np.array` or a `tf.Tensor`

        Returns
        -------
        log_lik     : tf.Tensor of shape (batch_size, n_obs)
            the approximate log-likelihood of each data point in each data set
        """

        return self.amortized_likelihood.log_likelihood(input_dict, to_numpy=to_numpy, **kwargs)
   
    def log_posterior(self, input_dict, to_numpy=True, **kwargs):
        """ Calculates the approximate log-posterior of targets given conditional variables via
        the change-of-variable formula for conditional normalizing flows.

        Parameters
        ----------
        input_dict : dict  
            Input dictionary containing the following mandatory keys, if DEFAULT_KEYS unchanged:
            `parameters`         - the latent generative model parameters over which a condition density is learned
            `summary_conditions` - the conditioning variables that are first passed through a summary network
            `direct_conditions`  - the conditioning variables that the directly passed to the inference network

        Returns
        -------
        log_post    : tf.Tensor of shape (batch_size, n_obs)
            the approximate log-likelihood of each data point in each data set
        """

        return self.amortized_posterior.log_posterior(input_dict, to_numpy=to_numpy, **kwargs)

    def log_marginal_likelihood(self, input_dict, to_numpy=True, **kwargs):
        """ Computes the approximate log marginal likelihood for some observed or simulated data.
        #TODO
        """

        pass

    def elpd(self, input_dict, to_numpy=True, **kwargs):
        """ Computes the expected log predictive density (ELPD) of some observed or simulated data.
        #TODO
        """

        pass
   
    def sample_data(self, input_dict, n_samples, to_numpy=True, **kwargs):
        """ Generates `n_samples` random draws from the surrogate likelihood given input conditions.

        Parameters
        ----------

        input_dict   : dict  
            Input dictionary containing the following mandatory keys, if DEFAULT_KEYS unchanged: 
            `conditions` - the conditioning variables that the directly passed to the inference network
        n_samples    : int
            The number of posterior samples to obtain from the approximate posterior
        to_numpy     : bool, optional, default: True
            Flag indicating whether to return the samples as a `np.array` or a `tf.Tensor`

        Returns
        -------
        lik_samples : tf.Tensor or np.ndarray of shape (n_datasets, n_samples, None)
            Simulated observables from the surrogate likelihood.
        """
        return self.amortized_likelihood.sample(input_dict, n_samples, to_numpy=to_numpy, **kwargs)

    def sample_parameters(self, input_dict, n_samples, to_numpy=True, **kwargs):
        """ Generates random draws from the approximate posterior given conditonal variables.

        Parameters
        ----------
        input_dict   : dict  
            Input dictionary containing the following mandatory keys, if DEFAULT KEYS unchanged: 
            `summary_conditions` : the conditioning variables (including data) that are first passed through a summary network
            `direct_conditions`  : the conditioning variables that the directly passed to the inference network
        n_samples    : int
            The number of posterior samples to obtain from the approximate posterior
        to_numpy     : bool, optional, default: True
            Boolean flag indicating whether to return the samples as a `np.array` or a `tf.Tensor`

        Returns
        -------
        post_samples : tf.Tensor or np.ndarray of shape (n_datasets, n_samples, n_params)
            the sampled parameters per data set
        """
        
        return self.amortized_posterior.sample(input_dict, n_samples, to_numpy=to_numpy, **kwargs)


class AmortizedModelComparer(tf.keras.Model):
    """ An interface to connect an evidential network for Bayesian model comparison with an optional summary network,
    as described in the original paper on evidential neural networks for model comparison:

    Radev, S. T., D'Alessandro, M., Mertens, U. K., Voss, A., Köthe, U., & Bürkner, P. C. (2021). 
    Amortized bayesian model comparison with evidential deep learning. 
    IEEE Transactions on Neural Networks and Learning Systems.

    Note: the original paper does not distinguish between the summary and the evidential networks, but
    treats them as a whole, with the appropriate architetcure dictated by the model application. For the
    sake of consistency, the BayesFlow library distinguisahes the two modules.
    """

    def __init__(self, evidential_net, summary_net=None, loss_fun=None):
        """Initializes a composite neural architecture for amortized bayesian model comparison.

        Parameters
        ----------
        evidential_net    : tf.keras.Model
            A neural network which outputs model evidences. 
        summary_net       : tf.keras.Model or None, optional, default: None
            An optional summary network
        loss_fun          : callable or None, optional, default: None
            The loss function which accepts the outputs of the amortizer. If None, the loss is inferred
            based on the `inference_net` type. 

        Important
        ----------
        - If no `summary_net` is provided, then the output dictionary of your generative model should not contain
        any `sumamry_conditions`, i.e., `summary_conditions` should be set to None, otherwise these will be ignored.

        - If no custom `loss_fun` is provided, the loss function will be the log loss for the means of a Dirichlet
        distribution, as described in:

        Radev, S. T., D'Alessandro, M., Mertens, U. K., Voss, A., Köthe, U., & Bürkner, P. C. (2021). 
        Amortized bayesian model comparison with evidential deep learning. 
        IEEE Transactions on Neural Networks and Learning Systems.

        The regularization weight will be set to 0.01
        """

        self.evidential_net = evidential_net
        self.summary_net = summary_net
        self.loss = self._determine_loss(loss_fun)

    def __call__(self, input_dict, return_summary=False, **kwargs):
        """ Performs a forward pass through both networks.

        Parameters
        ----------
        input_dict     : dict 
            Input dictionary containing the following mandatory keys, if DEFAULT_KEYS unchanged
            `summary_conditions` - the conditioning variables that are first passed through a summary network
            `direct_conditions`  - the conditioning variables that the directly passed to the evidential network
        return_summary : bool, optional, default: False
            Indicates whether the summary network outputs are returned along the estimated evidences.

        Returns
        -------
        #TODO
        """

        summary_out, full_cond = self._compute_summary_condition(
            input_dict.get(DEFAULT_KEYS['summary_conditions']), 
            input_dict.get(DEFAULT_KEYS['direct_conditions']),
            **kwargs
        )

        net_out = self.evidence_net(full_cond, **kwargs)

        if not return_summary:
            return net_out
        return net_out, summary_out

    def compute_loss(self, input_dict, **kwargs):
        pass

    def sample(self, input_dict, to_numpy=True, **kwargs):
        """Samples posterior model probabilities from the higher order Dirichlet density.

        Parameters
        ----------
        input_dict : dict
            Input dictionary containing the following mandatory keys, if DEFAULT_KEYS unchanged
            `summary_conditions` - the conditioning variables that are first passed through a summary network
            `direct_conditions`  - the conditioning variables that the directly passed to the evidential network
        n_samples  : int
            Number of samples to obtain from the approximate posterior
        to_numpy   : bool, default: True
            Flag indicating whether to return the samples as a np.array or a tf.Tensor
            
        Returns
        -------
        pm_samples : tf.Tensor or np.array
            The posterior draws from the Dirichlet distribution, shape (n_samples, n_batch, n_models)
        """

        _, full_cond = self._compute_summary_condition(
            input_dict.get(DEFAULT_KEYS['summary_conditions']), 
            input_dict.get(DEFAULT_KEYS['direct_conditions']),
            **kwargs
        )

        return self.evidence_net.sample(full_cond, to_numpy, **kwargs)

    def evidence(self, input_dict, to_numpy=True, **kwargs):
        """TODO"""

        _, full_cond = self._compute_summary_condition(
            input_dict.get(DEFAULT_KEYS['summary_conditions']), 
            input_dict.get(DEFAULT_KEYS['direct_conditions']),
            **kwargs
        )

        alphas = self(full_cond, return_summary=False, **kwargs)
        if to_numpy:
            return alphas.numpy()
        return alphas

    def uncertainty_score(self, input_dict, to_numpy=True, **kwargs):
        """TODO"""

        _, full_cond = self._compute_summary_condition(
            input_dict.get(DEFAULT_KEYS['summary_conditions']), 
            input_dict.get(DEFAULT_KEYS['direct_conditions']),
            **kwargs
        )

        alphas = self(full_cond, return_summary=False, **kwargs)
        u = tf.reduce_sum(alphas, axis=-1) / self.evidential_net.n_models
        if to_numpy:
            return u.numpy()
        return u

    def _compute_summary_condition(self, summary_conditions, direct_conditions, **kwargs):
        """ Determines how to concatenate the provided conditions.
        """

        # Compute learnable summaries, if given
        if self.summary_net is not None:
            sum_condition = self.summary_net(summary_conditions, **kwargs)
        else:
            sum_condition = None

        # Concatenate learnable summaries with fixed summaries, this 
        if sum_condition is not None and direct_conditions is not None:
            full_cond = tf.concat([sum_condition, direct_conditions], axis=-1)
        elif sum_condition is not None:
            full_cond = sum_condition
        elif direct_conditions is not None:
            full_cond = direct_conditions
        else:
            raise SummaryStatsError("Could not concatenarte or determine conditioning inputs...")
        return sum_condition, full_cond

    def _determine_loss(self, loss_fun):
        pass