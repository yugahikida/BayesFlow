import keras
import keras.ops as ops
from keras.saving import register_keras_serializable as serializable

from bayesflow.types import Tensor
from .mab import MultiHeadAttentionBlock


@serializable(package="bayesflow.networks")
class InducedSetAttentionBlock(keras.Layer):
    """Implements the ISAB block from [1] which represents learnable self-attention specifically
    designed to deal with large sets via a learnable set of "inducing points".

    [1] Lee, J., Lee, Y., Kim, J., Kosiorek, A., Choi, S., & Teh, Y. W. (2019).
        Set transformer: A framework for attention-based permutation-invariant neural networks.
        In International conference on machine learning (pp. 3744-3753). PMLR.
    """

    def __init__(
        self,
        num_inducing_points: int,
        key_dim: int = 32,
        num_heads: int = 4,
        dropout: float = 0.05,
        num_dense_feedforward: int = 2,
        output_dim: int = None,
        dense_units: int = 128,
        dense_activation: str = "gelu",
        kernel_initializer: str = "he_normal",
        use_bias=True,
        layer_norm: bool = True,
        **kwargs,
    ):
        """Creates a self-attention attention block with inducing points (ISAB) which will typically
        be used as part of a set transformer architecture according to [1].

        Parameters
        ----------
        #TODO
        """

        super().__init__(**kwargs)

        self.num_inducing_points = num_inducing_points
        self.inducing_points = self.add_weight(
            shape=(self.num_inducing_points, output_dim if output_dim is not None else key_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        mab_kwargs = dict(
            key_dim=key_dim,
            num_heads=num_heads,
            dropout=dropout,
            num_dense_feedforward=num_dense_feedforward,
            output_dim=output_dim,
            dense_units=dense_units,
            dense_activation=dense_activation,
            kernel_initializer=kernel_initializer,
            use_bias=use_bias,
            layer_norm=layer_norm,
        )
        self.mab0 = MultiHeadAttentionBlock(**mab_kwargs)
        self.mab1 = MultiHeadAttentionBlock(**mab_kwargs)

    def call(self, set_x: Tensor, **kwargs) -> Tensor:
        """Performs the forward pass through the self-attention layer.

        Parameters
        ----------
        set_x   : Tensor
            Input of shape (batch_size, set_size, input_dim)
            Since this is self-attention, the input set is used
            as a query (Q), key (K), and value (V)

        Returns
        -------
        out : Tensor
            Output of shape (batch_size, set_size, input_dim)
        """

        batch_size = ops.shape(set_x)[0]
        inducing_points_expanded = ops.expand_dims(self.inducing_points, axis=0)
        inducing_points_tiled = ops.tile(inducing_points_expanded, [batch_size, 1, 1])
        h = self.mab0(inducing_points_tiled, set_x, **kwargs)
        return self.mab1(set_x, h, **kwargs)
