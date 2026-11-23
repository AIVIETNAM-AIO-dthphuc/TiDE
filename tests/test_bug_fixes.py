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

"""Tests for --fix_final_decoder_ln and --fix_dropout."""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tempfile
import unittest
from pathlib import Path

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
  """One batch with unknown static shapes, as produced by data_loader."""
  rng = np.random.default_rng(0)

  def gen():
    yield (
        rng.normal(size=(_B, _L)).astype('f4'),
        rng.normal(size=(2, _L)).astype('f4'),
        np.zeros((1, _L), 'i4'),
        rng.normal(size=(_B, _H)).astype('f4'),
        rng.normal(size=(2, _H)).astype('f4'),
        np.zeros((1, _H), 'i4'),
        np.arange(_B, dtype='i4'),
    )

  types = tuple([tf.float32] * 2 + [tf.int32] + [tf.float32] * 2
                + [tf.int32] * 2)
  batch = next(iter(tf.data.Dataset.from_generator(gen, types)))
  return batch[:3], batch[4:6], batch[-1], batch[3]


def _model(fix_ln=True, fix_dropout=True, layer_norm=True, dropout_rate=0.5):
  tf.random.set_seed(0)
  return models.TideModel(
      _CONFIG, _H, [3], _NUM_TS, transform=True, layer_norm=layer_norm,
      dropout_rate=dropout_rate, fix_final_decoder_ln=fix_ln,
      fix_dropout=fix_dropout)


def _grad_norms(model, past, fut, tsidx, ytrue):
  """{variable name: max |grad|} of the MSE loss."""
  with tf.GradientTape() as tape:
    pred = model((past, fut, tsidx), training=False)
    loss = tf.reduce_mean(tf.square(pred - ytrue))
  grads = tape.gradient(loss, model.trainable_variables)
  return {
      v.name: 0.0 if g is None
      else float(tf.reduce_max(tf.abs(tf.convert_to_tensor(g))))
      for v, g in zip(model.trainable_variables, grads)
  }


def _train_losses(model, build_first, steps=3):
  """train_step losses with lr=0, so only dropout can change them."""
  past, fut, tsidx, ytrue = _inputs()
  if build_first:  # Same order as train._build_model_and_optimizer.
    model((past, fut, tsidx), training=False)
  opt = tf.keras.optimizers.SGD(0.0)
  return [float(model.train_step(past, fut, ytrue, tsidx, opt))
          for _ in range(steps)]


def _eval_outputs(model, steps=3):
  past, fut, tsidx, _ = _inputs()
  return [model((past, fut, tsidx), training=False).numpy()
          for _ in range(steps)]


class TestFinalDecoderLayerNorm(unittest.TestCase):

  def test_defaults_enable_both_fixes(self):
    model = models.TideModel(
        _CONFIG, _H, [3], _NUM_TS, transform=True, layer_norm=True)
    self.assertTrue(model.fix_final_decoder_ln)
    self.assertTrue(model.fix_dropout)
    self.assertFalse(model.final_decoder.layer_norm)

  def test_fix_disables_only_final_decoder_layer_norm(self):
    model = _model(fix_ln=True)
    self.assertFalse(model.final_decoder.layer_norm)
    self.assertFalse(hasattr(model.final_decoder, 'lnorm'))
    for block in (model.encoder, model.decoder, model.time_encoder):
      for layer in block.layers:
        self.assertTrue(layer.layer_norm)

  def test_original_bug_dead_backbone(self):
    past, fut, tsidx, ytrue = _inputs()
    model = _model(fix_ln=False, fix_dropout=False)
    self.assertTrue(model.final_decoder.layer_norm)
    grads = _grad_norms(model, past, fut, tsidx, ytrue)
    linear = {v.name for v in model.linear.trainable_variables}
    alive = {k for k, v in grads.items()
             if v > 0 and k not in linear and 'affine' not in k
             and 'beta' not in k}
    self.assertEqual(alive, set())

  def test_fix_restores_backbone_gradients(self):
    past, fut, tsidx, ytrue = _inputs()
    model = _model(fix_ln=True)
    grads = _grad_norms(model, past, fut, tsidx, ytrue)
    for block in (model.encoder, model.decoder, model.time_encoder,
                  model.final_decoder, model.ts_embs, *model.cat_embs):
      names = [v.name for v in block.trainable_variables]
      self.assertTrue(names)
      self.assertTrue(any(grads[n] > 0 for n in names), names)
    self.assertTrue(all(np.isfinite(v) for v in grads.values()))

  def test_ln_fix_is_noop_without_layer_norm(self):
    past, fut, tsidx, _ = _inputs()
    a = _model(fix_ln=False, layer_norm=False)
    b = _model(fix_ln=True, layer_norm=False)
    pa = a((past, fut, tsidx), training=False)
    b((past, fut, tsidx), training=False)
    b.set_weights(a.get_weights())
    pb = b((past, fut, tsidx), training=False)
    np.testing.assert_allclose(pa.numpy(), pb.numpy(), rtol=1e-6, atol=1e-6)


class TestDropout(unittest.TestCase):

  def _assert_varies(self, values):
    self.assertGreater(len({round(v, 6) for v in values}), 1, values)

  def _assert_constant(self, values):
    self.assertEqual(len({round(v, 6) for v in values}), 1, values)

  def _assert_eval_deterministic(self, model):
    outs = _eval_outputs(model)
    for out in outs[1:]:
      np.testing.assert_array_equal(outs[0], out)

  def test_fix_dropout_active_in_training_after_inference_build(self):
    model = _model(fix_dropout=True)
    self._assert_varies(_train_losses(model, build_first=True))
    self._assert_eval_deterministic(model)

  def test_fix_dropout_original_call_order(self):
    model = _model(fix_dropout=True)
    self._assert_varies(_train_losses(model, build_first=False))
    self._assert_eval_deterministic(model)

  def test_no_fix_reproduces_dropout_never_applied(self):
    model = _model(fix_dropout=False)
    self._assert_constant(_train_losses(model, build_first=True))
    self._assert_eval_deterministic(model)

  def test_no_fix_equals_zero_dropout_model(self):
    past, fut, tsidx, _ = _inputs()
    a = _model(fix_dropout=False, dropout_rate=0.5)
    b = _model(fix_dropout=True, dropout_rate=0.0)
    pa = a((past, fut, tsidx), training=True)
    b((past, fut, tsidx), training=True)
    b.set_weights(a.get_weights())
    pb = b((past, fut, tsidx), training=True)
    np.testing.assert_allclose(pa.numpy(), pb.numpy(), rtol=1e-6, atol=1e-6)

  def test_zero_rate_training_is_deterministic(self):
    model = _model(fix_dropout=True, dropout_rate=0.0)
    self._assert_constant(_train_losses(model, build_first=True))

  def test_train_step_updates_backbone_with_both_fixes(self):
    past, fut, tsidx, ytrue = _inputs()
    model = _model()
    model((past, fut, tsidx), training=False)
    before = [v.numpy().copy() for v in model.encoder.trainable_variables]
    loss = model.train_step(past, fut, ytrue, tsidx,
                            tf.keras.optimizers.Adam(1e-3))
    self.assertTrue(np.isfinite(float(loss)))
    after = [v.numpy() for v in model.encoder.trainable_variables]
    self.assertTrue(any(not np.allclose(b, a) for b, a in zip(before, after)))


class TestFlagsAndRunConfig(unittest.TestCase):
  _KEYS = ('fix_final_decoder_ln', 'fix_dropout')

  def setUp(self):
    super().setUp()
    if not FLAGS.is_parsed():
      FLAGS(['test_bug_fixes'])
    self._saved = {k: getattr(FLAGS, k) for k in self._KEYS + ('run_id',)}
    FLAGS.run_id = None

  def tearDown(self):
    for k, v in self._saved.items():
      setattr(FLAGS, k, v)
    super().tearDown()

  def _set(self, ln, dropout):
    FLAGS.fix_final_decoder_ln = ln
    FLAGS.fix_dropout = dropout

  def test_flag_defaults_true(self):
    for key in self._KEYS:
      self.assertTrue(FLAGS[key].default, key)

  def test_run_config_keys_only_when_enabled(self):
    for ln in (False, True):
      for dropout in (False, True):
        with self.subTest(ln=ln, dropout=dropout):
          self._set(ln, dropout)
          config = train._run_config()
          self.assertEqual('fix_final_decoder_ln' in config, ln)
          self.assertEqual('fix_dropout' in config, dropout)

  def _resume(self, saved, current):
    with tempfile.TemporaryDirectory() as tmp:
      folder = Path(tmp)
      self._set(*saved)
      train._verify_or_write_config(folder, train._run_config())
      self._set(*current)
      train._verify_or_write_config(folder, train._run_config())

  def test_resume_same_settings_accepted(self):
    for ln in (False, True):
      for dropout in (False, True):
        self._resume((ln, dropout), (ln, dropout))

  def test_resume_changed_settings_rejected(self):
    cases = [((False, False), (True, False), 'fix_final_decoder_ln'),
             ((True, False), (False, False), 'fix_final_decoder_ln'),
             ((False, False), (False, True), 'fix_dropout'),
             ((False, True), (False, False), 'fix_dropout')]
    for saved, current, key in cases:
      with self.subTest(saved=saved, current=current):
        with self.assertRaisesRegex(ValueError, key):
          self._resume(saved, current)

  def test_pre_fix_config_rejected_by_default_flags(self):
    """A run_config written before this branch has neither key."""
    with tempfile.TemporaryDirectory() as tmp:
      folder = Path(tmp)
      self._set(False, False)
      train._verify_or_write_config(folder, train._run_config())
      self._set(True, True)
      with self.assertRaisesRegex(ValueError, 'fix_dropout'):
        train._verify_or_write_config(folder, train._run_config())

  def test_mismatch_reports_key_with_null_value(self):
    with tempfile.TemporaryDirectory() as tmp:
      folder = Path(tmp)
      config = train._run_config()
      train._verify_or_write_config(folder, dict(config, extra_nullable=None))
      with self.assertRaisesRegex(ValueError, 'extra_nullable'):
        train._verify_or_write_config(folder, config)


if __name__ == '__main__':
  unittest.main()
