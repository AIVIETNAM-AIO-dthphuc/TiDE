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

"""Unit tests for Slice-level Adaptive Normalization (SAN)."""

import math
import unittest

from absl import flags
import models
import numpy as np
import tensorflow as tf
from tensorflow import keras
import train

FLAGS = flags.FLAGS


class TestSanHelpers(unittest.TestCase):
  """Tests for module-level SAN helper functions."""

  def setUp(self):
    super().setUp()
    tf.random.set_seed(42)
    np.random.seed(42)

  def test_slice_normalize_denormalize_round_trip(self):
    """Round-trip normalization and de-normalization reproduces input (atol 1e-4)."""
    batch_size = 16
    seq_len = 720
    slice_len = 24
    x = tf.random.normal((batch_size, seq_len), mean=10.0, stddev=3.0)

    x_norm, mu, sd = models.slice_normalize(x, slice_len)
    x_rec = models.slice_denormalize(x_norm, mu, sd, slice_len)

    np.testing.assert_allclose(x.numpy(), x_rec.numpy(), atol=1e-4)

    # Test with target sequence y and its true future statistics.
    pred_len = 96
    y = tf.random.normal((batch_size, pred_len), mean=-5.0, stddev=2.0)
    y_norm, mu_f, sd_f = models.slice_normalize(y, slice_len)
    y_rec = models.slice_denormalize(y_norm, mu_f, sd_f, slice_len)
    np.testing.assert_allclose(y.numpy(), y_rec.numpy(), atol=1e-4)

  def test_constant_slices_finite(self):
    """Constant slices with std 0 must yield finite outputs without NaN or Inf."""
    batch_size = 8
    seq_len = 720
    slice_len = 24
    # All elements in each slice are constant.
    x = tf.ones((batch_size, seq_len), dtype=tf.float32) * 7.5

    x_norm, mu, sd = models.slice_normalize(x, slice_len)
    self.assertTrue(np.isfinite(x_norm.numpy()).all())
    self.assertTrue(np.isfinite(mu.numpy()).all())
    self.assertTrue(np.isfinite(sd.numpy()).all())
    np.testing.assert_allclose(x_norm.numpy(), 0.0, atol=1e-5)

    x_rec = models.slice_denormalize(x_norm, mu, sd, slice_len)
    self.assertTrue(np.isfinite(x_rec.numpy()).all())
    np.testing.assert_allclose(x.numpy(), x_rec.numpy(), atol=1e-4)

  def test_extract_slice_stats(self):
    """extract_slice_stats computes correct slice-level mean and std."""
    batch_size = 4
    seq_len = 96
    slice_len = 24
    x = tf.random.normal((batch_size, seq_len))

    mu, sd = models.extract_slice_stats(x, slice_len)
    self.assertEqual(mu.shape, (batch_size, seq_len // slice_len))
    self.assertEqual(sd.shape, (batch_size, seq_len // slice_len))

    # Compare with numpy calculation per slice.
    x_np = x.numpy().reshape((batch_size, -1, slice_len))
    np_mu = np.mean(x_np, axis=-1)
    np_sd = np.std(x_np, axis=-1)
    np.testing.assert_allclose(mu.numpy(), np_mu, atol=1e-5)
    np.testing.assert_allclose(sd.numpy(), np_sd, atol=1e-5)


class TestSanStatsNetwork(unittest.TestCase):
  """Tests for SanStatsNetwork."""

  def setUp(self):
    super().setUp()
    tf.random.set_seed(42)
    np.random.seed(42)

  def test_std_head_strictly_positive(self):
    """Std head output must be strictly positive (> 0)."""
    batch_size = 8
    p_slices = 30
    f_slices = 4
    net = models.SanStatsNetwork(num_future_slices=f_slices)

    mu_p = tf.random.normal((batch_size, p_slices))
    sd_p = tf.random.uniform((batch_size, p_slices), minval=0.0, maxval=5.0)
    mean_x = tf.random.normal((batch_size, 1))

    _, sd_hat = net(mu_p, sd_p, mean_x)
    self.assertEqual(sd_hat.shape, (batch_size, f_slices))
    self.assertTrue((sd_hat.numpy() > 0).all())

  def test_untrained_mean_residual(self):
    """Untrained zero-initialized mean head residual predicts future mean = past mean."""
    batch_size = 8
    p_slices = 30
    f_slices = 4
    net = models.SanStatsNetwork(num_future_slices=f_slices)

    mu_p = tf.random.normal((batch_size, p_slices))
    sd_p = tf.random.uniform((batch_size, p_slices), minval=0.1, maxval=2.0)
    mean_x = tf.random.normal((batch_size, 1))

    mu_hat, _ = net(mu_p, sd_p, mean_x)
    self.assertEqual(mu_hat.shape, (batch_size, f_slices))
    expected_mu = np.repeat(mean_x.numpy(), f_slices, axis=1)
    np.testing.assert_allclose(mu_hat.numpy(), expected_mu, atol=1e-6)


class TestTideModelSan(unittest.TestCase):
  """Tests for TideModel with SAN enabled and disabled."""

  def setUp(self):
    super().setUp()
    tf.random.set_seed(42)
    np.random.seed(42)
    self.model_config = {
        'model_type': 'dnn',
        'hidden_dims': [64, 64],
        'time_encoder_dims': [64, 4],
        'decoder_output_dim': 4,
        'final_decoder_hidden': 64,
    }
    self.num_ts = 5
    self.cat_sizes = [3]

  def _make_dummy_inputs(self, batch_size, hist_len, pred_len):
    past_ts = tf.random.normal((batch_size, hist_len))
    past_feats = tf.random.normal((2, hist_len))
    past_cfeats = tf.zeros((1, hist_len), dtype=tf.int32)
    past_data = (past_ts, past_feats, past_cfeats)

    fut_feats = tf.random.normal((2, pred_len))
    fut_cfeats = tf.zeros((1, pred_len), dtype=tf.int32)
    future_features = (fut_feats, fut_cfeats)

    tsidx = tf.zeros((batch_size,), dtype=tf.int32)
    ytrue = tf.random.normal((batch_size, pred_len))
    return past_data, future_features, tsidx, ytrue

  def test_shapes_configurations(self):
    """Verifies shapes for (L, H, T) in {(720, 96, 24), (336, 720, 24), (720, 720, 24)}."""
    configs = [
        (720, 96, 24),
        (336, 720, 24),
        (720, 720, 24),
    ]
    batch_size = 4
    for hist_len, pred_len, slice_len in configs:
      with self.subTest(hist_len=hist_len, pred_len=pred_len, slice_len=slice_len):
        model = models.TideModel(
            model_config=self.model_config,
            pred_len=pred_len,
            cat_sizes=self.cat_sizes,
            num_ts=self.num_ts,
            san=True,
            san_slice_len=slice_len,
            san_lambda=0.5,
        )
        past_data, future_features, tsidx, _ = self._make_dummy_inputs(
            batch_size, hist_len, pred_len
        )

        out = model((past_data, future_features, tsidx), training=False)
        self.assertEqual(out.shape, (batch_size, pred_len))

        forecast, mu_hat, sd_hat = model._forward(
            (past_data, future_features, tsidx), training=False
        )
        self.assertEqual(forecast.shape, (batch_size, pred_len))
        self.assertEqual(mu_hat.shape, (batch_size, pred_len // slice_len))
        self.assertEqual(sd_hat.shape, (batch_size, pred_len // slice_len))

  def test_train_step_loss_and_gradients(self):
    """train_step with san=True returns MSE + lambda * stat_loss and updates with finite grads."""
    hist_len, pred_len, slice_len = 720, 96, 24
    batch_size = 4
    san_lambda = 0.5
    model = models.TideModel(
        model_config=self.model_config,
        pred_len=pred_len,
        cat_sizes=self.cat_sizes,
        num_ts=self.num_ts,
        san=True,
        san_slice_len=slice_len,
        san_lambda=san_lambda,
    )
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    past_data, future_features, tsidx, ytrue = self._make_dummy_inputs(
        batch_size, hist_len, pred_len
    )

    # Compute expected loss on the exact same predictions/statistics before update.
    preds, mu_hat, sd_hat = model._forward(
        (past_data, future_features, tsidx), training=True
    )
    pred_mse = tf.reduce_mean(tf.square(ytrue - preds))
    mu_f, sd_f = models.extract_slice_stats(ytrue, slice_len)
    stat_mse = tf.reduce_mean(tf.square(mu_f - mu_hat)) + tf.reduce_mean(
        tf.square(sd_f - sd_hat)
    )
    expected_total_loss = pred_mse + san_lambda * stat_mse

    # Execute train_step.
    actual_loss = model.train_step(
        past_data, future_features, ytrue, tsidx, optimizer
    )

    np.testing.assert_allclose(
        actual_loss.numpy(), expected_total_loss.numpy(), rtol=1e-4
    )

    # Check that all trainable variables remain finite after gradient update.
    for var in model.trainable_variables:
      self.assertTrue(
          np.isfinite(var.numpy()).all(),
          f'Variable {var.name} has non-finite values.',
      )

  def test_train_step_lambda_zero(self):
    """With san_lambda=0, train_step returns plain MSE loss."""
    hist_len, pred_len, slice_len = 720, 96, 24
    batch_size = 4
    model = models.TideModel(
        model_config=self.model_config,
        pred_len=pred_len,
        cat_sizes=self.cat_sizes,
        num_ts=self.num_ts,
        san=True,
        san_slice_len=slice_len,
        san_lambda=0.0,
    )
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    past_data, future_features, tsidx, ytrue = self._make_dummy_inputs(
        batch_size, hist_len, pred_len
    )

    preds, _, _ = model._forward(
        (past_data, future_features, tsidx), training=True
    )
    expected_mse = tf.reduce_mean(tf.square(ytrue - preds))

    actual_loss = model.train_step(
        past_data, future_features, ytrue, tsidx, optimizer
    )
    np.testing.assert_allclose(actual_loss.numpy(), expected_mse.numpy(), rtol=1e-4)

  def test_baseline_compatibility_san_false(self):
    """When san=False, model has no SAN layers and train_step returns plain MSE."""
    hist_len, pred_len = 720, 96
    batch_size = 4
    model = models.TideModel(
        model_config=self.model_config,
        pred_len=pred_len,
        cat_sizes=self.cat_sizes,
        num_ts=self.num_ts,
        transform=False,
        san=False,
    )
    past_data, future_features, tsidx, ytrue = self._make_dummy_inputs(
        batch_size, hist_len, pred_len
    )

    _ = model((past_data, future_features, tsidx), training=False)

    # Check no SAN variables exist.
    for var in model.trainable_variables:
      self.assertNotIn('san_stats_network', var.name)
      self.assertNotIn('affine', var.name)

    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    preds = model((past_data, future_features, tsidx), training=True)
    expected_loss = tf.reduce_mean(tf.square(ytrue - preds))
    actual_loss = model.train_step(
        past_data, future_features, ytrue, tsidx, optimizer
    )
    np.testing.assert_allclose(actual_loss.numpy(), expected_loss.numpy(), rtol=1e-4)


class TestTrainFlagsAndValidation(unittest.TestCase):
  """Tests for train.py flags, _resolve_arguments, and _run_config."""

  def setUp(self):
    super().setUp()
    if not FLAGS.is_parsed():
      FLAGS(['test_san'])
    # Set default valid base flags.
    FLAGS.dataset = 'elec'
    FLAGS.random_seed = 42
    FLAGS.seed = None
    FLAGS.horizon = None
    FLAGS.pred_len = 96
    FLAGS.hist_len = 720
    FLAGS.san = False
    FLAGS.san_slice_len = 24
    FLAGS.san_lambda = 0.5
    FLAGS.run_id = None

  def test_run_config_san_false(self):
    """_run_config does not contain SAN keys when san=False."""
    FLAGS.san = False
    config = train._run_config()
    self.assertNotIn('san', config)
    self.assertNotIn('san_slice_len', config)
    self.assertNotIn('san_lambda', config)

  def test_run_config_san_true(self):
    """_run_config contains SAN keys when san=True."""
    FLAGS.san = True
    FLAGS.san_slice_len = 24
    FLAGS.san_lambda = 0.5
    config = train._run_config()
    self.assertIn('san', config)
    self.assertIn('san_slice_len', config)
    self.assertIn('san_lambda', config)
    self.assertEqual(config['san'], True)
    self.assertEqual(config['san_slice_len'], 24)
    self.assertEqual(config['san_lambda'], 0.5)

  def test_resolve_arguments_valid(self):
    """_resolve_arguments succeeds with valid SAN parameters."""
    FLAGS.san = True
    FLAGS.hist_len = 720
    FLAGS.pred_len = 96
    FLAGS.san_slice_len = 24
    FLAGS.san_lambda = 0.5
    dataset = train._resolve_arguments()
    self.assertEqual(dataset, 'electricity')

  def test_resolve_arguments_invalid_slice_len(self):
    """_resolve_arguments raises ValueError if T <= 0."""
    FLAGS.san = True
    FLAGS.san_slice_len = 0
    with self.assertRaises(ValueError):
      train._resolve_arguments()

    FLAGS.san_slice_len = -5
    with self.assertRaises(ValueError):
      train._resolve_arguments()

  def test_resolve_arguments_indivisible_hist_len(self):
    """_resolve_arguments raises ValueError if hist_len % T != 0."""
    FLAGS.san = True
    FLAGS.hist_len = 336
    FLAGS.san_slice_len = 96
    FLAGS.pred_len = 96
    with self.assertRaises(ValueError):
      train._resolve_arguments()

  def test_resolve_arguments_indivisible_pred_len(self):
    """_resolve_arguments raises ValueError if pred_len % T != 0."""
    FLAGS.san = True
    FLAGS.hist_len = 720
    FLAGS.san_slice_len = 50
    FLAGS.pred_len = 96
    with self.assertRaises(ValueError):
      train._resolve_arguments()

  def test_resolve_arguments_invalid_lambda(self):
    """_resolve_arguments raises ValueError if san_lambda < 0, nan, or inf."""
    FLAGS.san = True
    FLAGS.san_slice_len = 24
    FLAGS.hist_len = 720
    FLAGS.pred_len = 96

    FLAGS.san_lambda = -0.1
    with self.assertRaises(ValueError):
      train._resolve_arguments()

    FLAGS.san_lambda = float('nan')
    with self.assertRaises(ValueError):
      train._resolve_arguments()

    FLAGS.san_lambda = float('inf')
    with self.assertRaises(ValueError):
      train._resolve_arguments()

  def test_resolve_arguments_san_false_skips_validation(self):
    """When san=False, indivisible hist_len/T or invalid lambda does not raise."""
    FLAGS.san = False
    FLAGS.hist_len = 336
    FLAGS.san_slice_len = 96
    FLAGS.san_lambda = -1.0
    FLAGS.pred_len = 96
    dataset = train._resolve_arguments()
    self.assertEqual(dataset, 'electricity')


if __name__ == '__main__':
  unittest.main()
