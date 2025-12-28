# tests/test_deephalo.py

import tensorflow as tf
import pytest

from DeepHalo import Phi, DeepHalo

#Test the linear trasnformation phi produces correct shape
def test_phi_output_shape():
    """Phi should map (B, d) → (B, H, embed)."""
    batch_size = 8
    d_in = 16
    H = 4
    embed = 10

    phi = Phi(H=H, embed=embed, dropout=0.1)
    x = tf.random.normal((batch_size, d_in))

    y = phi(x, training=True)

    assert y.shape == (batch_size, H, embed)


def test_deephalo_initialization():
    """Check that DeepHalo initializes expected number of layers."""
    model = DeepHalo(L=3, embedding_dim=32, H=4, dropout_rate=0.1)

    assert len(model.W_layers) == 3
    assert len(model.phi_layers) == 3
    assert model.heads == 4
    assert model.embedding_dim == 32


def test_compute_batch_utility_shapes():
    """compute_batch_utility should return (B, S) utilities."""
    batch_size = 5
    n_items = 3
    d_shared = 4
    d_item = 6

    model = DeepHalo(L=2, embedding_dim=8, H=3, dropout_rate=0.0)

    shared_feats = tf.random.normal((batch_size, d_shared))
    items_feats = tf.random.normal((batch_size, n_items, d_item))
    available = tf.ones((batch_size, n_items), dtype=tf.bool)
    choices = tf.zeros((batch_size,), dtype=tf.int32)

    utilities = model.compute_batch_utility(
        shared_features_by_choice=shared_feats,
        items_features_by_choice=items_feats,
        available_items_by_choice=available,
        choices=choices
    )

    assert utilities.shape == (batch_size, n_items)
    # Ensure utilities are finite
    assert tf.math.reduce_all(tf.math.is_finite(utilities))



#Integration test that shows backprop works and gradients are non‑zero—good
def test_deephalo_backpropagation():
    """
    Integration test: can compute gradients and apply one optimizer step.
    """
    batch_size = 4
    n_items = 3
    d_shared = 5
    d_item = 7

    model = DeepHalo(L=2, embedding_dim=16, H=4, dropout_rate=0.0)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    shared_feats = tf.random.normal((batch_size, d_shared))
    items_feats = tf.random.normal((batch_size, n_items, d_item))
    available = tf.ones((batch_size, n_items), dtype=tf.bool)
    # For a dummy loss, define some arbitrary "targets"
    # e.g., prefer item 0 in each choice set
    target_utilities = tf.ones((batch_size, n_items))
    target_utilities = tf.concat(
        [tf.ones((batch_size, 1)), tf.zeros((batch_size, n_items - 1))],
        axis=1,
    )

    with tf.GradientTape() as tape:
        utilities = model.compute_batch_utility(
            shared_features_by_choice=shared_feats,
            items_features_by_choice=items_feats,
            available_items_by_choice=available,
            choices=None
        )
        # Simple MSE loss just to test gradients
        loss = tf.reduce_mean(tf.square(utilities - target_utilities))

    grads = tape.gradient(loss, model.trainable_weights)
    # Ensure some gradients are not None and not all zeros
    non_none_grads = [g for g in grads if g is not None]
    assert len(non_none_grads) > 0
    assert any(tf.reduce_any(tf.not_equal(g, 0.0)) for g in non_none_grads)

    # Apply one optimization step
    optimizer.apply_gradients(zip(grads, model.trainable_weights))