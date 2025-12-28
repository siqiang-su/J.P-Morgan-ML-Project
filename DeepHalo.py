import numpy as np
import pandas as pd
from choice_learn.data import ChoiceDataset
from choice_learn.data.storage import FeaturesStorage
import tensorflow as tf
from choice_learn.models.base_model import ChoiceModel
from tensorflow.keras.layers import Dense, LayerNormalization, BatchNormalization, ReLU, Dropout

class Phi(tf.keras.layers.Layer):
    """
    Head-specific non-linear transform φ used in Zhang(25).

    This layer maps a base embedding z ∈ ℝ^d to H head-specific embeddings
    φ_h(z) ∈ ℝ^embed, h = 1, …, H.

    Parameters
    ----------
    H : int
        Number of heads.
    embed : int
        Dimensionality of the output embedding per head. Author sets it to be 128.
    dropout : float, optional
        Dropout rate applied after the first dense layer, by default 0.0.
    """
    def __init__(self, H, embed=128, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.H = H
        self.embed = embed

        # First linear projection to H * embed, followed by ReLU activation. Different weights for each head
        self.fc1 = Dense(embed * H, activation="relu")

        # Second linear projection back to embed per head, shared weights
        self.fc2 = Dense(embed)
        self.dropout = Dropout(dropout)
        self.layer_norm = LayerNormalization()
    

    def call(self, X, training=False):
        """
        Forward pass.

        Parameters
        ----------
        X : tf.Tensor
            Input tensor of shape (B, d) where B is batch size and d is item feature dim.
        training : bool, optional
            Whether the layer is in training mode (controls dropout), by default False.

        Returns
        -------
        tf.Tensor
            Tensor of shape (B, H, embed); head-specific embeddings.
        """
        B = tf.shape(X)[0]
        n = tf.shape(X)[1]
        # First linear layer and reshape
        X = self.fc1(X)  
        X = tf.reshape(X, (B, self.H, self.embed))  # (B, H, embed)
        X = self.dropout(X, training=training)
        X = self.fc2(X)  # (B, H, embed)
        X = self.layer_norm(X)
        return X

class DeepHalo(ChoiceModel):
    """
    Implementation of DeepHalo choice model.

    This model learns item embeddings with halo effects and maps them to utilities. This model is based on the base model in choice_learn package.

    Parameters
    ----------
    L : int
        Interaction order (number of halo interaction layers).
    embedding_dim : int
        Dimensionality of the internal item embedding.
    H : int
        Number of halo heads.
    dropout_rate : float, optional
        Dropout rate in the MLP and Phi layers, by default 0.0.
    batch_size : int, optional
        Default batch size for training, by default 256.
    kwargs :
        Additional arguments passed to ChoiceModel.
    """
    def __init__(self, L, embedding_dim, H, dropout_rate = 0, batch_size=256, **kwargs):
        super().__init__(**kwargs)
        self.heads = H
        #self.no_alters = S
        self.interaction_order = L
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.lr = 0.003

        # Shared features → embedding   
        self.shared_layer = Dense(
            units = embedding_dim,
            activation = 'relu',
            kernel_initializer='he_normal',
            name = 'shared'
        )

        # Item MLP: three layers mapping item features to base embedding z^0
        self.dense1 = Dense(
            units=embedding_dim,
            activation='relu',  # We'll apply ReLU separately
            kernel_initializer='he_normal',
            name="dense1"
        )
        self.batch_norm1 = BatchNormalization()
        self.dropout1 = Dropout(self.dropout_rate)
        
        self.dense2 = Dense(
            units=embedding_dim,
            activation='relu',
            kernel_initializer='he_normal',
            name="dense2"
        )
        self.batch_norm2 = BatchNormalization()
        self.dropout2 = Dropout(self.dropout_rate)
        
        self.dense3 = Dense(
            units=self.embedding_dim,
            activation=None,
            kernel_initializer='he_normal',
            name="dense3"
        )

        self.layer_norm = LayerNormalization(epsilon=1e-6)

        # W^l layers that project embeddings to head scores, l reflects the totat interaction order L
        self.W_layers = [
            Dense(
                units=self.heads,
                activation=None,
                use_bias=False,
                kernel_initializer="he_normal",
                name=f"W_{l+1}"
            )
            for l in range(self.interaction_order)
        ]

        # φ^l non-linear transformation layers
        self.phi_layers = [
            Phi(H=self.heads, embed=self.embedding_dim,
                dropout=dropout_rate, name=f"phi_{l+1}")
            for l in range(self.interaction_order)
        ]

        # LayerNorm after the second layer
        self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1)

        # Final layer: embeddings to utility
        self.final_layer = Dense(units=1, activation="linear", use_bias = False)



    @property
    def trainable_weights(self):
        """
        Collect all trainable weights of the model.

        Returns
        -------
        list[tf.Variable]
            List of all trainable variables.
        """
        vars_W = [v
              for layer in self.W_layers
              for v in layer.trainable_variables]

    # From list of phi_layers
        vars_phi = [v
                    for layer in self.phi_layers
                    for v in layer.trainable_variables]
        return self.shared_layer.trainable_variables\
              + self.dense1.trainable_variables\
              + self.dense2.trainable_variables\
              + self.dense3.trainable_variables\
              + vars_phi\
              + vars_W\
                  + self.final_layer.trainable_variables\
                  + self.layer_norm.trainable_variables\
                  + self.batch_norm1.trainable_variables\
                  + self.batch_norm2.trainable_variables

    
    def compute_batch_utility(self,
                              shared_features_by_choice,
                              items_features_by_choice,
                              available_items_by_choice,
                              choices):
        """
        Compute item utilities for a batch of choice situations.

        Parameters
        ----------
        shared_features_by_choice : tf.Tensor
            Tensor of shape (B, d_shared) with context/shared features per choice.
        items_features_by_choice : tf.Tensor
            Tensor of shape (B, S, d_item) with item features for each choice set.
            B is batch size, S is number of items per choice.
        available_items_by_choice : tf.Tensor
            Boolean or {0,1} mask of shape (B, S) indicating available items.
        choices : tf.Tensor
            Tensor of shape (B,) with chosen item indices.
        training : bool, optional
            Whether the model is in training mode, by default False.

        Returns
        -------
        tf.Tensor
            Tensor of utilities of shape (B, S).
        """
        _, _ = available_items_by_choice, choices
        # We apply the neural network to all items_features_by_choice for all the items
        # We then concatenate the utilities of each item of shape (n_choices, 1) into a single one of shape (n_choices, n_items)
   

        shared_features_embedding = self.shared_layer(shared_features_by_choice)

        items_features_embeddings = []
        # Loop over items in the choice set
        if isinstance(items_features_by_choice, tf.Tensor):
            n_item = items_features_by_choice.shape[1]
        else:
            n_item = items_features_by_choice[0].shape[1]
        for i in range(n_item):
            # Utility is Dense(embeddings sum)
            x = self.dense1(items_features_by_choice[:, i])
            x = self.batch_norm1(x, training=True)
            x = self.dropout1(x, training=True)
            
            # Second Dense + BatchNorm + ReLU + Dropout
            x = self.dense2(x)
            x = self.batch_norm2(x, training=True)
            x = self.dropout2(x, training=True)
            
            # Final Dense + LayerNorm
            x = self.dense3(x)
            z0 = self.layer_norm(x) #(B, d0)

            #Z_bar component
            z_prev = z0

            # Halo interaction layers
            for l in range(self.interaction_order):
            # 1) Context summary: Z̄^l = 1/S Σ_k W^l z_k^{l-1}
                Z_context = self.W_layers[l](z_prev)        # (B, H)
                Z_bar_l = tf.reduce_mean(Z_context, axis=0) # (H, )

                # 2) Head-specific nonlinear transforms of base embedding z^0
                #    φ^l(z^0) -> (B, H, d)
                phi_out = self.phi_layers[l](z0, training=True)

                # 3) Weighted halo: (1/H) Σ_h Z̄_h^l * φ_h^l(z_j^0)
                Z_bar_exp = tf.expand_dims(Z_bar_l, axis=-1)  # (B, H, 1)
                weighted = Z_bar_exp * phi_out                     # (B, H, d)
                temp = tf.reduce_sum(weighted, axis=1) / self.heads  # (B, d)

                # 4) Residual update: z_j^l = z_j^{l-1} + temp
                z_prev = z_prev + temp                             # (B, d)

            # Combine with shared embedding and normalize
            z_L = z_prev + shared_features_embedding

            items_features_embeddings.append(self.final_layer(z_L))

        # Concatenation to get right shape (n_choices, n_items, )
        item_utility_by_choice = tf.concat(items_features_embeddings, axis=1)

        return item_utility_by_choice
    

class DeepHalo_Featureless(ChoiceModel):
    def __init__(self, L, embedding_dim, H, dropout_rate = 0, hidden_dims=None, batch_size=256, **kwargs):
        super().__init__(**kwargs)
        self.heads = H
        #self.no_alters = S
        self.interaction_order = L
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.lr = 0.01
        self.loss = tf.keras.losses.MeanSquaredError(
            reduction='sum_over_batch_size',  
            name='mean_squared_error'
        )

        self.hidden_dims = hidden_dims

        #First, the function X map each feature to a new embedding via three-layer MLP with RelU activation
        self.input_norm = LayerNormalization(epsilon=1e-6)


        self.dense1 = Dense(
            units=embedding_dim,
            activation='elu',  # We'll apply ReLU separately
            kernel_initializer='he_normal',
            name="dense1"
        )
        self.batch_norm1 = BatchNormalization()
        self.dropout1 = Dropout(self.dropout_rate)
        
        # Second hidden layer: hidden_dims[0] -> hidden_dims[1]
        self.dense2 = Dense(
            units=embedding_dim,
            activation='elu',
            kernel_initializer='he_normal',
            name="dense2"
        )
        self.batch_norm2 = BatchNormalization()
        self.dropout2 = Dropout(self.dropout_rate)
        
        # Final layer: hidden_dims[1] -> embedding_dim
        self.dense3 = Dense(
            units=self.embedding_dim,
            activation=None,
            kernel_initializer='he_normal',
            name="dense3"
        )

        self.layer_norm = LayerNormalization(epsilon=1e-6)


        self.W_layers = [
            Dense(
                units=self.heads,
                activation=None,
                use_bias=False,
                kernel_initializer="he_normal",
                name=f"W_{l+1}"
            )
            for l in range(self.interaction_order)
        ]

        self.phi_layers = [
            Phi(H=self.heads, embed=self.embedding_dim,
                dropout=dropout_rate, name=f"phi_{l+1}")
            for l in range(self.interaction_order)
        ]

        #After we generate the embedding z^0, we linearly aggregating the result to from Z_1 bar:
        self.Z1_bar = Dense(
            units = self.heads,
            activation = None,
            use_bias = False,
            kernel_initializer = 'he_normal',
            name = 'Z1bar'
        )

        #self.phi_layers = [Phi(H=self.heads, embed=self.embedding_dim) for _ in range(self.heads)]
        self.phi = Phi(H=self.heads, embed=self.embedding_dim)


        # LayerNorm after the second layer
        self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1)

        # Third layer: embeddings to utility (dense representation of features > U)
        self.final_layer = Dense(units=1, activation="linear", use_bias = False)

    # We do not forget to specify self.trainable_weights with all coefficients that need to be estimated.
    # Small trick using @property to acces future weights of layers
    # that have not been instantiated yet !
    @property
    def trainable_weights(self):
        """Endpoint to acces model's trainable_weights.

        Returns:
        --------
        list
            list of trainable_weights
        """
        vars_W = [v
              for layer in self.W_layers
              for v in layer.trainable_variables]

    # From list of phi_layers
        vars_phi = [v
                    for layer in self.phi_layers
                    for v in layer.trainable_variables]
        return self.dense1.trainable_variables\
              + self.dense2.trainable_variables\
              + self.dense3.trainable_variables\
              + vars_phi\
              + vars_W\
                  + self.final_layer.trainable_variables\
                  + self.layer_norm.trainable_variables\
                  + self.batch_norm1.trainable_variables\
                  + self.batch_norm2.trainable_variables

    
    def compute_batch_utility(self,
                              shared_features_by_choice,
                              items_features_by_choice,
                              available_items_by_choice,
                              choices):
        """Computes batch utility from features."""
        # We apply the neural network to all items_features_by_choice for all the items
        # We then concatenate the utilities of each item of shape (n_choices, 1) into a single one of shape (n_choices, n_items)

        items_features_embeddings = []
        for i in range(items_features_by_choice[0].shape[1]):
            # Utility is Dense(embeddings sum)
            x = self.dense1(items_features_by_choice[:, i])
            x = self.batch_norm1(x, training=True)
            x = self.dropout1(x, training=True)
            
            # Second Dense + BatchNorm + ReLU + Dropout
            x = self.dense2(x)
            x = self.batch_norm2(x, training=True)
            x = self.dropout2(x, training=True)
            
            # Final Dense + LayerNorm
            x = self.dense3(x)
            z0 = self.layer_norm(x) #(B, d0)

            #Z_bar component
            z_prev = z0

            for l in range(self.interaction_order):
            # 1) Context summary: Z̄^l = 1/S Σ_k W^l z_k^{l-1}
            #    W^l(z_prev) -> (B, S, H)
                Z_context = self.W_layers[l](z_prev)        # (B, H)
                Z_bar_l = tf.reduce_mean(Z_context, axis=0) # (H, )

                # 2) Head-specific nonlinear transforms of base embedding z^0
                #    φ^l(z^0) -> (B, S, H, d)
                phi_out = self.phi_layers[l](z0, training=True)

                # 3) Weighted halo: (1/H) Σ_h Z̄_h^l * φ_h^l(z_j^0)
                Z_bar_exp = tf.expand_dims(Z_bar_l, axis=-1)  # (B, H, 1)
                weighted = Z_bar_exp * phi_out                     # (B, H, d)
                temp = tf.reduce_sum(weighted, axis=1) / self.heads  # (B, S, d)

                # 4) Residual update: z_j^l = z_j^{l-1} + temp
                z_prev = z_prev + temp                             # (B, S, d)

            z_L = z_prev

            items_features_embeddings.append(self.final_layer(z_L))

        # Concatenation to get right shape (n_choices, n_items, )
        item_utility_by_choice = tf.concat(items_features_embeddings, axis=1)

        return item_utility_by_choice