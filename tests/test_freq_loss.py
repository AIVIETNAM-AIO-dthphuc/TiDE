# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

"""Tests for frequency-enhanced loss and TiDE integration."""

import unittest
from absl import flags
import numpy as np
import tensorflow as tf
from tensorflow import keras

import models
import train

FLAGS = flags.FLAGS


class FrequencyEnhancedLossTest(unittest.TestCase):
  """Unit tests for freq_enhanced_loss and TideModel integration."""

  @classmethod
  def setUpClass(cls):
    if not FLAGS.is_parsed():
      FLAGS.mark_as_parsed()

  def test_alpha_zero_equals_mse(self):
    """alpha=0 equals keras MeanSquaredError (abs diff <= 1e-6) across horizons."""
    mse_fn = keras.losses.MeanSquaredError()
    for horizon in (96, 192, 336, 720):
      y_true = tf.random.normal((8, horizon), dtype=tf.float32, seed=42)
      y_pred = tf.random.normal((8, horizon), dtype=tf.float32, seed=43)
      expected_mse = mse_fn(y_true, y_pred)
      loss = models.freq_enhanced_loss(y_true, y_pred, alpha=0.0)
      abs_diff = tf.abs(loss - expected_mse).numpy()
      self.assertLessEqual(
          abs_diff,
          1e-6,
          f'Failed for horizon {horizon}: abs_diff={abs_diff}',
      )

  def test_alpha_one_matches_numpy_reference(self):
    """alpha=1 matches NumPy rfft L1 reference (rel diff <= 1e-5) across horizons."""
    for horizon in (96, 192, 336, 720):
      np.random.seed(100 + horizon)
      y_true_np = np.random.randn(8, horizon).astype(np.float32)
      y_pred_np = np.random.randn(8, horizon).astype(np.float32)

      fft_true_np = np.fft.rfft(y_true_np, axis=-1)
      fft_pred_np = np.fft.rfft(y_pred_np, axis=-1)
      expected_np = np.mean(np.abs(fft_pred_np - fft_true_np))

      y_true_tf = tf.constant(y_true_np)
      y_pred_tf = tf.constant(y_pred_np)
      loss_tf = models.freq_enhanced_loss(y_true_tf, y_pred_tf, alpha=1.0).numpy()

      rel_diff = abs(loss_tf - expected_np) / expected_np
      self.assertLessEqual(
          rel_diff,
          1e-5,
          f'Failed for horizon {horizon}: rel_diff={rel_diff}',
      )

  def test_large_offset_numerical_stability(self):
    """Residual FFT avoids catastrophic cancellation on large offsets (H=720, alpha=1.0)."""
    horizon = 720
    y_pred_np = np.full((1, horizon), 10000.01, dtype=np.float64)
    y_true_np = np.full((1, horizon), 10000.0, dtype=np.float64)
    expected_np = np.mean(
        np.abs(np.fft.rfft(y_pred_np - y_true_np, axis=-1))
    )

    y_pred_tf = tf.constant(y_pred_np)
    y_true_tf = tf.constant(y_true_np)
    loss_tf = models.freq_enhanced_loss(y_true_tf, y_pred_tf, alpha=1.0).numpy()

    rel_diff = abs(loss_tf - expected_np) / expected_np
    self.assertLessEqual(
        rel_diff,
        1e-3,
        f'Failed numerical stability check: rel_diff={rel_diff}',
    )

  def test_alpha_half_weighted_combination(self):
    """alpha=0.5 equals 0.5 * MSE + 0.5 * freq_loss across horizons."""
    for horizon in (96, 336, 720):
      y_true = tf.random.normal((8, horizon), dtype=tf.float32, seed=50)
      y_pred = tf.random.normal((8, horizon), dtype=tf.float32, seed=51)
      loss_0 = models.freq_enhanced_loss(y_true, y_pred, alpha=0.0)
      loss_1 = models.freq_enhanced_loss(y_true, y_pred, alpha=1.0)
      expected_combined = 0.5 * loss_0 + 0.5 * loss_1
      loss_half = models.freq_enhanced_loss(y_true, y_pred, alpha=0.5)
      abs_diff = tf.abs(loss_half - expected_combined).numpy()
      self.assertLessEqual(
          abs_diff,
          1e-6,
          f'Failed for horizon {horizon}: abs_diff={abs_diff}',
      )

  def test_identical_inputs_loss_zero(self):
    """y_pred == y_true -> loss is 0 for any alpha."""
    for horizon in (96, 720):
      y = tf.random.normal((8, horizon), dtype=tf.float32, seed=60)
      for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        loss = models.freq_enhanced_loss(y, y, alpha=alpha)
        self.assertAlmostEqual(
            float(loss.numpy()),
            0.0,
            places=6,
            msg=f'Failed for alpha={alpha}, horizon={horizon}',
        )

  def test_finite_gradients_including_identical_inputs(self):
    """Gradients w.r.t. y_pred are finite (no NaN/Inf), even when y_pred == y_true."""
    for horizon in (96, 720):
      for alpha in (0.0, 0.5, 1.0):
        # Distinct inputs
        y_true = tf.random.normal((4, horizon), dtype=tf.float32, seed=70)
        y_pred = tf.Variable(
            tf.random.normal((4, horizon), dtype=tf.float32, seed=71)
        )
        with tf.GradientTape() as tape:
          loss = models.freq_enhanced_loss(y_true, y_pred, alpha=alpha)
        grad = tape.gradient(loss, y_pred)
        self.assertFalse(
            np.any(np.isnan(grad.numpy())),
            f'NaN grad for distinct inputs alpha={alpha}, H={horizon}',
        )
        self.assertFalse(
            np.any(np.isinf(grad.numpy())),
            f'Inf grad for distinct inputs alpha={alpha}, H={horizon}',
        )

        # Identical inputs (y_pred == y_true)
        y_pred_identical = tf.Variable(tf.identity(y_true))
        with tf.GradientTape() as tape:
          loss_ident = models.freq_enhanced_loss(
              y_true, y_pred_identical, alpha=alpha
          )
        grad_ident = tape.gradient(loss_ident, y_pred_identical)
        self.assertFalse(
            np.any(np.isnan(grad_ident.numpy())),
            f'NaN grad for identical inputs alpha={alpha}, H={horizon}',
        )
        self.assertFalse(
            np.any(np.isinf(grad_ident.numpy())),
            f'Inf grad for identical inputs alpha={alpha}, H={horizon}',
        )

  def test_tide_model_stores_alpha(self):
    """TideModel stores the freq_loss_alpha passed to it, defaulting to 0.0."""
    model_config = {
        'hidden_dims': [64],
        'time_encoder_dims': [32, 4],
        'decoder_output_dim': 2,
        'final_decoder_hidden': 16,
    }
    model_default = models.TideModel(
        model_config=model_config,
        pred_len=96,
        cat_sizes=[],
        num_ts=1,
    )
    self.assertEqual(model_default.freq_loss_alpha, 0.0)

    model_custom = models.TideModel(
        model_config=model_config,
        pred_len=96,
        cat_sizes=[],
        num_ts=1,
        freq_loss_alpha=0.75,
    )
    self.assertEqual(model_custom.freq_loss_alpha, 0.75)

  def test_tide_model_train_step(self):
    """TideModel.train_step uses freq_enhanced_loss with the model's alpha."""
    model_config = {
        'hidden_dims': [32],
        'time_encoder_dims': [16, 4],
        'decoder_output_dim': 2,
        'final_decoder_hidden': 8,
    }
    batch, hist_len, pred_len, num_ts = 4, 16, 96, 2
    past_data = (
        tf.random.normal((batch, hist_len), seed=1),
        tf.random.normal((2, hist_len), seed=2),
        tf.zeros((0, hist_len), dtype=tf.int32),
    )
    future_features = (
        tf.random.normal((2, pred_len), seed=3),
        tf.zeros((0, pred_len), dtype=tf.int32),
    )
    tsidx = tf.zeros((batch,), dtype=tf.int32)
    ytrue = tf.random.normal((batch, pred_len), seed=4)

    # 1. Test alpha = 0.5: loss must match freq_enhanced_loss(..., 0.5) and differ from MSE
    model_half = models.TideModel(
        model_config=model_config,
        pred_len=pred_len,
        cat_sizes=[],
        num_ts=num_ts,
        dropout_rate=0.0,
        freq_loss_alpha=0.5,
    )
    optimizer_half = keras.optimizers.Adam(1e-3)
    preds_half = model_half((past_data, future_features, tsidx), training=True)
    expected_loss_half = models.freq_enhanced_loss(ytrue, preds_half, alpha=0.5)
    plain_mse_half = keras.losses.MeanSquaredError()(ytrue, preds_half)

    loss_half = model_half.train_step(
        past_data, future_features, ytrue, tsidx, optimizer_half
    )
    self.assertAlmostEqual(
        float(loss_half.numpy()),
        float(expected_loss_half.numpy()),
        places=5,
        msg='train_step loss must equal freq_enhanced_loss with alpha=0.5',
    )
    self.assertNotAlmostEqual(
        float(loss_half.numpy()),
        float(plain_mse_half.numpy()),
        places=2,
        msg='train_step loss with alpha=0.5 must differ from plain MSE',
    )

    # 2. Test alpha = 0.0: loss must equal plain MSE
    model_zero = models.TideModel(
        model_config=model_config,
        pred_len=pred_len,
        cat_sizes=[],
        num_ts=num_ts,
        dropout_rate=0.0,
        freq_loss_alpha=0.0,
    )
    optimizer_zero = keras.optimizers.Adam(1e-3)
    preds_zero = model_zero((past_data, future_features, tsidx), training=True)
    expected_mse_zero = keras.losses.MeanSquaredError()(ytrue, preds_zero)

    loss_zero = model_zero.train_step(
        past_data, future_features, ytrue, tsidx, optimizer_zero
    )
    self.assertAlmostEqual(
        float(loss_zero.numpy()),
        float(expected_mse_zero.numpy()),
        places=5,
        msg='train_step loss with alpha=0.0 must equal plain MSE',
    )

  def test_run_config_compatibility(self):
    """_run_config has no freq_loss_alpha at alpha=0, but contains it at alpha!=0."""
    FLAGS.random_seed = 1
    FLAGS.pred_len = 96
    FLAGS.dataset = 'elec'

    FLAGS.freq_loss_alpha = 0.0
    cfg0 = train._run_config()
    self.assertNotIn('freq_loss_alpha', cfg0)

    FLAGS.freq_loss_alpha = 0.5
    cfg_half = train._run_config()
    self.assertIn('freq_loss_alpha', cfg_half)
    self.assertEqual(cfg_half['freq_loss_alpha'], 0.5)

    FLAGS.freq_loss_alpha = 1.0
    cfg_one = train._run_config()
    self.assertIn('freq_loss_alpha', cfg_one)
    self.assertEqual(cfg_one['freq_loss_alpha'], 1.0)

    # Reset back to default
    FLAGS.freq_loss_alpha = 0.0

  def test_freq_loss_alpha_validation(self):
    """Values of freq_loss_alpha outside [0, 1] or non-finite raise ValueError in _resolve_arguments."""
    FLAGS.random_seed = 1
    FLAGS.pred_len = 96
    FLAGS.dataset = 'elec'

    # Valid values should not raise
    for valid_alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
      FLAGS.freq_loss_alpha = valid_alpha
      train._resolve_arguments()

    # Invalid values must raise ValueError: out of range and non-finite
    for invalid_alpha in (
        -0.5, -0.01, 1.01, 2.0,
        float('nan'), float('inf'), float('-inf'),
    ):
      FLAGS.freq_loss_alpha = invalid_alpha
      with self.assertRaises(ValueError):
        train._resolve_arguments()

    # Reset back to default
    FLAGS.freq_loss_alpha = 0.0


if __name__ == '__main__':
  unittest.main()
