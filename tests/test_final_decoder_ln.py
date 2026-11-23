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

"""Tests for the --fix_final_decoder_ln option."""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import unittest

from absl import flags
import models
import numpy as np
import tensorflow as tf
import train

FLAGS = flags.FLAGS

_CONFIG = {
    'model_type': 'dnn',
    'hidden_dims': [64, 64],
    'time_encoder_dims': [64, 4],
    'decoder_output_dim': 4,
    'final_decoder_hidden': 64,
}
_B, _L, _H, _NUM_TS = 5, 48, 24, 5


def _inputs():
  rng = np.random.default_rng(0)
  past = (
      tf.constant(rng.normal(size=(_B, _L)), tf.float32),
      tf.constant(rng.normal(size=(2, _L)), tf.float32),
      tf.zeros((1, _L), tf.int32),
  )
  fut = (
      tf.constant(rng.normal(size=(2, _H)), tf.float32),
      tf.zeros((1, _H), tf.int32),
  )
  tsidx = tf.range(_B)
  ytrue = tf.constant(rng.normal(size=(_B, _H)), tf.float32)
  return past, fut, tsidx, ytrue


def _backbone_grad_norms(model, past, fut, tsidx, ytrue):
  """Returns {variable name: max |grad|} for variables outside the linear path."""
  with tf.GradientTape() as tape:
    pred = model((past, fut, tsidx), training=False)
    loss = tf.reduce_mean(tf.square(pred - ytrue))
  grads = tape.gradient(loss, model.trainable_variables)
  linear_ids = {id(v) for v in model.linear.trainable_variables}
  out = {}
  for var, grad in zip(model.trainable_variables, grads):
    if id(var) in linear_ids or 'affine' in var.name:
      continue
    if 'san_stats_network' in var.name:
      continue
    out[var.name] = (
        0.0 if grad is None
        else float(tf.reduce_max(tf.abs(tf.convert_to_tensor(grad)))))
  return out


class TestFinalDecoderLayerNorm(unittest.TestCase):

  def _model(self, fix, layer_norm=True, **kwargs):
    tf.random.set_seed(0)
    return models.TideModel(
        _CONFIG, _H, [3], _NUM_TS, transform=True, layer_norm=layer_norm,
        fix_final_decoder_ln=fix, **kwargs)

  def test_default_keeps_original_layer_norm(self):
    model = self._model(fix=False)
    self.assertFalse(model.fix_final_decoder_ln)
    self.assertTrue(model.final_decoder.layer_norm)
    self.assertTrue(model.encoder.layers[0].layer_norm)

  def test_fix_disables_only_final_decoder_layer_norm(self):
    model = self._model(fix=True)
    self.assertFalse(model.final_decoder.layer_norm)
    self.assertFalse(hasattr(model.final_decoder, 'lnorm'))
    for block in (model.encoder, model.decoder, model.time_encoder):
      for layer in block.layers:
        self.assertTrue(layer.layer_norm)

  def test_fix_is_noop_without_layer_norm(self):
    past, fut, tsidx, _ = _inputs()
    a = self._model(fix=False, layer_norm=False)
    b = self._model(fix=True, layer_norm=False)
    pa = a((past, fut, tsidx), training=False)
    b((past, fut, tsidx), training=False)
    self.assertEqual(
        [v.shape for v in a.trainable_variables],
        [v.shape for v in b.trainable_variables])
    b.set_weights(a.get_weights())
    pb = b((past, fut, tsidx), training=False)
    np.testing.assert_allclose(pa.numpy(), pb.numpy(), rtol=1e-6, atol=1e-6)

  def test_original_bug_constant_final_decoder_and_dead_backbone(self):
    past, fut, tsidx, ytrue = _inputs()
    model = self._model(fix=False)
    model((past, fut, tsidx), training=False)
    out = model.final_decoder(tf.random.normal((_B, _H, 8), seed=1))
    self.assertEqual(len(np.unique(out.numpy())), 1)
    grads = _backbone_grad_norms(model, past, fut, tsidx, ytrue)
    alive = {k: v for k, v in grads.items() if v > 0 and 'beta' not in k}
    self.assertEqual(alive, {})

  def _check_backbone_alive(self, **kwargs):
    past, fut, tsidx, ytrue = _inputs()
    model = self._model(fix=True, **kwargs)
    model((past, fut, tsidx), training=False)
    out = model.final_decoder(tf.random.normal((_B, _H, 8), seed=1))
    self.assertGreater(len(np.unique(out.numpy())), 1)
    grads = _backbone_grad_norms(model, past, fut, tsidx, ytrue)
    for prefix in ('encoder', 'decoder', 'time_encoder', 'final_decoder'):
      block = getattr(model, prefix)
      kernels = [v.name for v in block.trainable_variables
                 if 'kernel' in v.name]
      self.assertTrue(kernels)
      self.assertTrue(
          any(grads[name] > 0 for name in kernels),
          f'{prefix} receives no gradient')
    for value in grads.values():
      self.assertTrue(np.isfinite(value))

  def test_fix_restores_backbone_gradients(self):
    self._check_backbone_alive()

  def test_fix_restores_backbone_gradients_with_san(self):
    self._check_backbone_alive(san=True, san_slice_len=12)

  def test_train_step_with_fix_updates_backbone(self):
    past, fut, tsidx, ytrue = _inputs()
    model = self._model(fix=True, san=True, san_slice_len=12)
    model((past, fut, tsidx), training=False)
    before = [v.numpy().copy() for v in model.encoder.trainable_variables]
    opt = tf.keras.optimizers.Adam(1e-3)
    loss = model.train_step(past, fut, ytrue, tsidx, opt)
    self.assertTrue(np.isfinite(float(loss)))
    after = [v.numpy() for v in model.encoder.trainable_variables]
    self.assertTrue(any(not np.allclose(b, a) for b, a in zip(before, after)))


class TestFixFlag(unittest.TestCase):

  def setUp(self):
    super().setUp()
    if not FLAGS.is_parsed():
      FLAGS(['test_final_decoder_ln'])
    self._saved = {k: getattr(FLAGS, k)
                   for k in ('fix_final_decoder_ln', 'san', 'run_id')}
    FLAGS.san = False
    FLAGS.run_id = None

  def tearDown(self):
    for k, v in self._saved.items():
      setattr(FLAGS, k, v)
    super().tearDown()

  def test_flag_default_false(self):
    self.assertFalse(FLAGS['fix_final_decoder_ln'].default)

  def test_run_config_without_fix_has_no_key(self):
    FLAGS.fix_final_decoder_ln = False
    self.assertNotIn('fix_final_decoder_ln', train._run_config())

  def test_run_config_with_fix_has_key(self):
    FLAGS.fix_final_decoder_ln = True
    config = train._run_config()
    self.assertIn('fix_final_decoder_ln', config)
    self.assertTrue(config['fix_final_decoder_ln'])


if __name__ == '__main__':
  unittest.main()
