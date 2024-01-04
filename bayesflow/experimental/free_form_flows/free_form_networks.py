# Copyright (c) 2022 The BayesFlow Developers

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import tensorflow as tf
from typing import TypedDict, Callable
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Add

class ModelHParams(TypedDict):
    data_dim: int
    cond_dim: int
    latent_dim: int
    
class FFFHParams(ModelHParams):
    num_dense_layers: int
    num_resnet_layers: int
    units: int
    activation: str

class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, units, activation='silu', batch_norm=False):
        super(ResidualBlock, self).__init__()
        self.batch_norm1 = batch_norm
        self.activation = activation
        self.batch_norm = batch_norm

        self.dense1 = Dense(units, kernel_initializer='he_normal')
        self.batch_norm1 = BatchNormalization() if batch_norm is not False else None
        self.dense2 = Dense(units, activation=None)  # No activation here because we'll add the input
        self.batch_norm2 = BatchNormalization() if batch_norm is not False else None

    def call(self, inputs):
        x = self.dense1(inputs)
        if self.batch_norm is not False:
            x = self.batch_norm1(x)
        x = Activation(self.activation)(x)

        x = self.dense2(x)
        if self.batch_norm is not False:
            x = self.batch_norm2(x)
        x = Add()([x, inputs]) 
        x = Activation(self.activation)(x)  
        return x
        
class FFFNetwork(tf.keras.Model):
    def __init__(self, hparams: ModelHParams):
        super().__init__()
        self.hparams = hparams
        self.data_dim = hparams['data_dim']
        self.cond_dim = hparams['cond_dim']
        self.latent_dim = hparams['latent_dim']

        self.e_blocks = []
        self.d_blocks = []
        for i in range(hparams['num_dense_layers']):
            self.e_blocks.append(Dense(hparams['units'], activation=hparams['activation']))
            self.d_blocks.append(Dense(hparams['units'], activation=hparams['activation']))
        for i in range(hparams['num_resnet_layers']):
            self.e_blocks.append(ResidualBlock(hparams['units'], activation=hparams['activation']))
            self.d_blocks.append(ResidualBlock(hparams['units'], activation=hparams['activation']))
        self.e_blocks.append(Dense(self.latent_dim, activation='linear'))
        self.d_blocks.append(Dense(self.data_dim, activation='linear'))

    def call(self, targets, condition, inverse=False):
        output = tf.concat([targets, condition], -1)
        if inverse:
            for block in self.d_blocks:
                output = block(output)
            return output
        else:
            for block in self.e_blocks:
                output = block(output)
            return output
        
class FreeFormWrapper(tf.keras.Model):
    """Implements a chain of conditional invertible coupling layers for conditional density estimation."""

    def __init__(
        self,
        encoder: tf.keras.Model,
        decoder: tf.keras.Model,
        latent_dim: int,
        reshape_fct_encoder: Callable = None,
        reshape_fct_decoder: Callable = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim

        self.reshape_fct_encoder = reshape_fct_encoder
        self.reshape_fct_decoder = reshape_fct_decoder

    def call(self, targets, condition, inverse=False, **kwargs):
        if inverse:
            return self.inverse(targets, condition, **kwargs)
        return self.forward(targets, condition, **kwargs)

    def forward(self, targets, condition, **kwargs):
        """Performs a forward pass though the chain."""
        input = tf.concat([targets, condition], -1)
        if self.reshape_fct_encoder is not None:
            input = self.reshape_fct_encoder(input)
        z = self.encoder(input, **kwargs)
        return z

    def inverse(self, z, condition, **kwargs):
        """Performs a reverse pass through the chain. Assumes that it is only used
        in inference mode, so ``**kwargs`` contains ``training=False``."""
        input = tf.concat([z, condition], -1)
        if self.reshape_fct_decoder is not None:
            input = self.reshape_fct_decoder(input)
        x = self.decoder(input, **kwargs)
        return x