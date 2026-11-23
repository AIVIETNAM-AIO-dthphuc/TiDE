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

"""TiDE training with epoch-boundary TensorFlow checkpoints and optional rclone backup.

Compatible with the supplied drive_backup.py and the original TiDE absl flags.
Example (run from the TiDE source directory after configuring rclone):

  python train.py --dataset=elec --hist_len=720 --pred_len=336 \\
      --random_seed=1 --drive_backup=true

Equivalent convenience flags: --dataset=electricity --horizon=336 --seed=1.

Resume is automatic when --resume=true (default); independent fresh runs
should specify a unique --run_id and --resume=false.
"""

import json
import math
import os
from pathlib import Path
import random
import re
import sys

from absl import app
from absl import flags
from absl import logging
import data_loader
import models
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

FLAGS = flags.FLAGS

# Original TiDE flags; leave the optimization / dataset defaults unchanged.
flags.DEFINE_integer('train_epochs', 100, 'Number of epochs to train (original loop includes epoch 0)')
flags.DEFINE_integer('patience', 40, 'Patience for early stopping')
flags.DEFINE_integer('epoch_len', None, 'Number of iterations in an epoch')
flags.DEFINE_integer('batch_size', 512, 'Batch size for the randomly sampled batch')
flags.DEFINE_float('learning_rate', 1e-4, 'Learning rate')
flags.DEFINE_string('expt_dir', './results', 'Optional TensorBoard root (legacy flag)')
flags.DEFINE_string('dataset', 'elec', 'Dataset key (elec or electricity, traffic, etc.)')
flags.DEFINE_string('datetime_col', 'date', 'Column containing datetimes')
flags.DEFINE_list('num_cov_cols', None, 'Numerical covariate columns')
flags.DEFINE_list('cat_cov_cols', None, 'Categorical covariate columns')
flags.DEFINE_integer('hist_len', 720, 'Lookback/history length')
flags.DEFINE_integer('pred_len', 720, 'Forecasting horizon')
flags.DEFINE_integer('num_layers', 2, 'Number of DNN layers')
flags.DEFINE_integer('hidden_size', 256, 'Hidden size of DNN')
flags.DEFINE_integer('decoder_output_dim', 4, 'Decoder output dimension')
flags.DEFINE_integer('final_decoder_hidden', 64, 'Final decoder hidden dimension')
flags.DEFINE_list('ts_cols', None, 'Columns of time-series features')
flags.DEFINE_integer('random_seed', None, 'Random seed for TF and NumPy')
flags.DEFINE_bool('normalize', True, 'Normalize data for training')
flags.DEFINE_bool('holiday', False, 'Use holiday features')
flags.DEFINE_bool('permute', True, 'Permute series during training')
flags.DEFINE_bool('transform', False, 'Apply reversible transform')
flags.DEFINE_bool('layer_norm', False, 'Apply layer norm')
flags.DEFINE_bool(
    'fix_final_decoder_ln', True,
    'Disable layer norm in the final decoder (output dim 1), where it zeroes '
    'all backbone gradients when --layer_norm=true. Set false to reproduce '
    'the original (buggy) code and resume runs made before this fix.')
flags.DEFINE_float('dropout_rate', 0.0, 'Dropout rate')
flags.DEFINE_float('freq_loss_alpha', 0.0, 'Weight of frequency-enhanced loss in [0, 1]. 0.0 is baseline MSE.')
flags.DEFINE_integer('num_split', 1, 'Number of splits during inference')
flags.DEFINE_integer('min_num_epochs', 0, 'Minimum epochs before early stopping')
flags.DEFINE_integer('gpu', 0, 'GPU index; CPU is used if no GPU is available')

# Convenience arguments for notebook launches (aliases of original flags).
flags.DEFINE_integer('seed', None, 'Alias for --random_seed')
flags.DEFINE_integer('horizon', None, 'Alias for --pred_len')

# Checkpoint and Google Drive integration.
flags.DEFINE_bool('drive_backup', False, 'Enable rclone upload/restore to/from Drive')
flags.DEFINE_bool('resume', True, 'Resume the local or Google Drive checkpoint if found')
flags.DEFINE_integer('backup_every', 1, 'Upload once every N finished epochs')
flags.DEFINE_integer('max_to_keep', 2, 'Max local TF checkpoints retained')
flags.DEFINE_string('run_id', None, 'Optional unique ID for a separate independent run')
flags.DEFINE_string('local_base', '/kaggle/working/IS403_Experiment',
                    'Local experiment root for checkpoints and results')
flags.DEFINE_string('remote_base',
                    'gdrive:[54] IS403 - BA/[1] PROJECT/[IS403]_Experiment',
                    'Google Drive remote experiment root')

DATA_DICT = {
    'ettm2': {
        'boundaries': [34560, 46080, 57600],
        'data_path': './datasets/ETT-small/ETTm2.csv',
        'freq': '15min',
    },
    'ettm1': {
        'boundaries': [34560, 46080, 57600],
        'data_path': './datasets/ETT-small/ETTm1.csv',
        'freq': '15min',
    },
    'etth2': {
        'boundaries': [8640, 11520, 14400],
        'data_path': './datasets/ETT-small/ETTh2.csv',
        'freq': 'H',
    },
    'etth1': {
        'boundaries': [8640, 11520, 14400],
        'data_path': './datasets/ETT-small/ETTh1.csv',
        'freq': 'H',
    },
    'elec': {
        'boundaries': [18413, 21044, 26304],
        'data_path': './datasets/electricity/electricity.csv',
        'freq': 'H',
    },
    'traffic': {
        'boundaries': [12280, 14036, 17544],
        'data_path': './datasets/traffic/traffic.csv',
        'freq': 'H',
    },
    'weather': {
        'boundaries': [36887, 42157, 52696],
        'data_path': './datasets/weather/weather.csv',
        'freq': '10min',
    },
}


def _resolve_arguments():
  """Resolve notebook-friendly aliases without changing original TiDE flags."""
  if FLAGS.seed is not None:
    if FLAGS['random_seed'].present and FLAGS.random_seed != FLAGS.seed:
      raise ValueError('--seed and --random_seed disagree.')
    FLAGS.random_seed = FLAGS.seed
  if FLAGS.horizon is not None:
    if FLAGS['pred_len'].present and FLAGS.pred_len != FLAGS.horizon:
      raise ValueError('--horizon and --pred_len disagree.')
    FLAGS.pred_len = FLAGS.horizon

  FLAGS.dataset = FLAGS.dataset.lower()
  if FLAGS.dataset == 'electricity':
    FLAGS.dataset = 'elec'  # Official source name; Drive uses 'electricity'.
  if FLAGS.dataset not in DATA_DICT:
    raise ValueError(f'Unknown dataset {FLAGS.dataset!r}. Choose one of {list(DATA_DICT)}')
  if FLAGS.random_seed is None or FLAGS.random_seed <= 0:
    raise ValueError('Provide --random_seed=N (or --seed=N), with N > 0.')
  if FLAGS.pred_len not in (96, 192, 336, 720):
    raise ValueError('Horizon must be one of 96, 192, 336, 720.')
  if FLAGS.backup_every < 1 or FLAGS.max_to_keep < 1:
    raise ValueError('--backup_every and --max_to_keep must be positive integers.')
  if FLAGS.train_epochs < 0:
    raise ValueError('--train_epochs cannot be negative.')
  if FLAGS.run_id is not None and not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', FLAGS.run_id):
    raise ValueError('--run_id must contain 1-64 letters, digits, underscores or hyphens.')
  if (
      not math.isfinite(FLAGS.freq_loss_alpha)
      or FLAGS.freq_loss_alpha < 0.0
      or FLAGS.freq_loss_alpha > 1.0
  ):
    raise ValueError(
        f'--freq_loss_alpha must be a finite number in [0, 1], got {FLAGS.freq_loss_alpha}.'
    )
  return 'electricity' if FLAGS.dataset == 'elec' else FLAGS.dataset


def _run_config():
  """Fields that must not silently change when resuming a training run."""
  keys = (
      'dataset', 'random_seed', 'hist_len', 'pred_len', 'batch_size',
      'learning_rate', 'num_layers', 'hidden_size', 'decoder_output_dim',
      'final_decoder_hidden', 'datetime_col', 'ts_cols', 'num_cov_cols',
      'cat_cov_cols', 'epoch_len', 'normalize', 'holiday', 'permute',
      'transform', 'layer_norm', 'dropout_rate', 'num_split',
      'min_num_epochs', 'patience', 'train_epochs',
  )
  if FLAGS.fix_final_decoder_ln:
    keys += ('fix_final_decoder_ln',)
  config = {key: getattr(FLAGS, key) for key in keys}
  config['run_id'] = FLAGS.run_id
  if FLAGS.freq_loss_alpha != 0.0:
    config['freq_loss_alpha'] = FLAGS.freq_loss_alpha
  return config


def _verify_or_write_config(folder: Path, config: dict):
  """Reject incompatible checkpoints before loading any model/optimizer state."""
  config_file = folder / 'run_config.json'
  if config_file.exists():
    existing = json.loads(config_file.read_text(encoding='utf-8'))
    if existing != config:
      differences = sorted(
          k for k in set(config) | set(existing)
          if existing.get(k) != config.get(k))
      raise ValueError(
          f'Checkpoint configuration does not match current run: {differences}. '
          'Use the original hyperparameters or a NEW --run_id and --resume=false.'
      )
  else:
    # An existing TF checkpoint without metadata cannot be validated safely.
    if tf.train.latest_checkpoint(str(folder)):
      raise RuntimeError(f'Missing run_config.json for existing checkpoint: {folder}')
    folder.mkdir(parents=True, exist_ok=True)
    temp = folder / 'run_config.json.tmp'
    temp.write_text(json.dumps(config, indent=2, sort_keys=True), encoding='utf-8')
    os.replace(temp, config_file)


def _configure_device():
  gpus = tf.config.list_physical_devices('GPU')
  if not gpus:
    logging.warning('No GPU detected; training will run on CPU.')
    return
  if FLAGS.gpu < 0 or FLAGS.gpu >= len(gpus):
    raise ValueError(f'--gpu={FLAGS.gpu} invalid: found {len(gpus)} GPU(s).')
  tf.config.set_visible_devices(gpus[FLAGS.gpu], 'GPU')
  try:
    tf.config.experimental.set_memory_growth(gpus[FLAGS.gpu], True)
  except RuntimeError as error:
    logging.warning('GPU memory growth not changed: %s', error)


def _build_model_and_optimizer(model, optimizer, data):
  """Create model variables and Adam slots before TF checkpoint restore.

  TensorFlow/Keras 2.10 uses OptimizerV2 and does NOT expose build().
  Its _create_all_weights(var_list) creates iterations, hypers and Adam's
  momentum/variance slot variables without applying any gradient update.
  Keras optimizers introduced in newer versions may expose build().
  """
  sample = next(iter(data.tf_dataset(mode='train')))
  model((sample[:3], sample[4:6], sample[-1]), training=False)
  variables = model.trainable_variables
  if not variables:
    raise RuntimeError('Model has no trainable variables after initial forward pass.')
  if hasattr(optimizer, '_create_all_weights'):
    optimizer._create_all_weights(variables)  # TF 2.10 / Keras OptimizerV2
  elif hasattr(optimizer, 'build'):
    optimizer.build(variables)  # Newer Keras optimizers
  else:
    raise RuntimeError('Optimizer cannot initialize slots for checkpoint restore.')


def _save_best_outputs(expt_dir, val_res, test_res, test_metrics):
  expt_dir.mkdir(parents=True, exist_ok=True)
  with open(expt_dir / 'val_pred.npy', 'wb') as fp:
    np.save(fp, val_res[0][:, 0:-1:FLAGS.pred_len])
  with open(expt_dir / 'val_true.npy', 'wb') as fp:
    np.save(fp, val_res[1][:, 0:-1:FLAGS.pred_len])
  with open(expt_dir / 'test_pred.npy', 'wb') as fp:
    np.save(fp, test_res[0][:, 0:-1:FLAGS.pred_len])
  with open(expt_dir / 'test_true.npy', 'wb') as fp:
    np.save(fp, test_res[1][:, 0:-1:FLAGS.pred_len])
  with open(expt_dir / 'test_metrics.json', 'w', encoding='utf-8') as fp:
    json.dump({key: float(value) for key, value in test_metrics.items()}, fp, indent=2)


def training():
  """Official TiDE training loop with resumable end-of-epoch checkpoints."""
  drive_dataset = _resolve_arguments()
  _configure_device()
  tf.random.set_seed(FLAGS.random_seed)
  np.random.seed(FLAGS.random_seed)
  random.seed(FLAGS.random_seed)

  # Import does not initialize rclone; it is only required if enabled.
  from drive_backup import DriveBackup

  backup = None
  if FLAGS.drive_backup:
    backup = DriveBackup(
        dataset=drive_dataset,
        seed=FLAGS.random_seed,
        horizon=FLAGS.pred_len,
        run_id=FLAGS.run_id,
        local_base=FLAGS.local_base,
        remote_base=FLAGS.remote_base,
    )
    backup.check_connection()  # Fail before expensive training if Drive is unavailable.

  run_root = (
      Path(FLAGS.local_base) / drive_dataset /
      f'seed_{FLAGS.random_seed}' / f'H{FLAGS.pred_len}'
  )
  ckpt_dir = run_root / 'checkpoints'
  results_dir = run_root / 'results'
  if FLAGS.run_id:
    ckpt_dir = ckpt_dir / 'runs' / FLAGS.run_id
    results_dir = results_dir / 'runs' / FLAGS.run_id

  # Determine whether to resume. Never overwrite existing checkpoints silently.
  remote_info = None
  local_ckpt = tf.train.latest_checkpoint(str(ckpt_dir))
  if FLAGS.resume:
    if local_ckpt:
      logging.info('Using local checkpoint: %s', local_ckpt)
    elif backup:
      remote_info = backup.restore_latest_checkpoint(checkpoint_dir=ckpt_dir)
      logging.info('Remote restore: %s', remote_info or 'no existing snapshot')
  else:
    if local_ckpt:
      raise RuntimeError(
          f'Existing local checkpoint at {ckpt_dir}. '
          'Use --resume=true, or choose a unique --run_id.'
      )
    if backup and backup.list_completed_snapshots():
      raise RuntimeError(
          'A completed Drive snapshot already exists for this dataset/seed/horizon/run_id. '
          'Use --resume=true or choose a different --run_id.'
      )

  _verify_or_write_config(ckpt_dir, _run_config())
  results_dir.mkdir(parents=True, exist_ok=True)

  dataset = FLAGS.dataset
  data_path = DATA_DICT[dataset]['data_path']
  freq = DATA_DICT[dataset]['freq']
  boundaries = DATA_DICT[dataset]['boundaries']
  data_df = pd.read_csv(data_path)

  if FLAGS.ts_cols:
    # Keep the original TiDE behavior: optional dataset metadata is in DATA_DICT.
    ts_cols = DATA_DICT[dataset]['ts_cols']
    num_cov_cols = DATA_DICT[dataset]['num_cov_cols']
    cat_cov_cols = DATA_DICT[dataset]['cat_cov_cols']
  else:
    ts_cols = [col for col in data_df.columns if col != FLAGS.datetime_col]
    num_cov_cols = None
    cat_cov_cols = None

  dtl = data_loader.TimeSeriesdata(
      data_path=data_path,
      datetime_col=FLAGS.datetime_col,
      num_cov_cols=num_cov_cols,
      cat_cov_cols=cat_cov_cols,
      ts_cols=np.array(ts_cols),
      train_range=[0, boundaries[0]],
      val_range=[boundaries[0], boundaries[1]],
      test_range=[boundaries[1], boundaries[2]],
      hist_len=FLAGS.hist_len,
      pred_len=FLAGS.pred_len,
      batch_size=min(FLAGS.batch_size, len(ts_cols)),
      freq=freq,
      normalize=FLAGS.normalize,
      epoch_len=FLAGS.epoch_len,
      holiday=FLAGS.holiday,
      permute=FLAGS.permute,
  )

  model_config = {
      'model_type': 'dnn',
      'hidden_dims': [FLAGS.hidden_size] * FLAGS.num_layers,
      'time_encoder_dims': [64, 4],
      'decoder_output_dim': FLAGS.decoder_output_dim,
      'final_decoder_hidden': FLAGS.final_decoder_hidden,
      'batch_size': dtl.batch_size,
  }
  model = models.TideModel(
      model_config=model_config,
      pred_len=FLAGS.pred_len,
      num_ts=len(ts_cols),
      cat_sizes=dtl.cat_sizes,
      transform=FLAGS.transform,
      layer_norm=FLAGS.layer_norm,
      dropout_rate=FLAGS.dropout_rate,
      freq_loss_alpha=FLAGS.freq_loss_alpha,
      fix_final_decoder_ln=FLAGS.fix_final_decoder_ln,
  )

  # The decay schedule is unchanged from the supplied original code.
  lr_schedule = keras.optimizers.schedules.CosineDecay(
      initial_learning_rate=FLAGS.learning_rate,
      decay_steps=30 * dtl.train_range[1],
  )
  optimizer = keras.optimizers.Adam(learning_rate=lr_schedule, clipvalue=1e3)

  epoch_counter = tf.Variable(0, dtype=tf.int64, trainable=False, name='completed_epochs')
  best_rmse = tf.Variable(np.inf, dtype=tf.float64, trainable=False, name='best_rmse')
  patience_counter = tf.Variable(0, dtype=tf.int64, trainable=False, name='patience_counter')
  finished = tf.Variable(False, dtype=tf.bool, trainable=False, name='finished')

  _build_model_and_optimizer(model, optimizer, dtl)
  checkpoint = tf.train.Checkpoint(
      model=model,
      optimizer=optimizer,
      epoch=epoch_counter,
      best_rmse=best_rmse,
      patience=patience_counter,
      finished=finished,
  )
  manager = tf.train.CheckpointManager(
      checkpoint, directory=str(ckpt_dir), max_to_keep=FLAGS.max_to_keep
  )
  if FLAGS.resume and manager.latest_checkpoint:
    status = checkpoint.restore(manager.latest_checkpoint)
    # All model and optimizer variables were created before restore.
    status.assert_existing_objects_matched()
    logging.info('RESUMED: %s; completed_epochs=%d; best_rmse=%.6f; patience=%d',
                 manager.latest_checkpoint, int(epoch_counter.numpy()),
                 float(best_rmse.numpy()), int(patience_counter.numpy()))
    if remote_info is not None and remote_info['epoch'] != int(epoch_counter.numpy()):
      raise RuntimeError('Drive snapshot metadata epoch disagrees with TF checkpoint epoch.')

  logging.info(
      'Run: dataset=%s seed=%d H=%d freq_loss_alpha=%s fix_final_decoder_ln=%s '
      'local=%s remote=%s',
      drive_dataset, FLAGS.random_seed, FLAGS.pred_len, FLAGS.freq_loss_alpha,
      FLAGS.fix_final_decoder_ln,
      ckpt_dir, backup.remote_snapshots if backup else '[disabled]',
  )
  if bool(finished.numpy()):
    logging.info('This run is already marked complete. No further training performed.')
    return

  log_dir = results_dir / 'tensorboard'
  if FLAGS['expt_dir'].present:
    log_dir = Path(FLAGS.expt_dir) / f'{drive_dataset}_seed{FLAGS.random_seed}_H{FLAGS.pred_len}'
  summary = Summary(str(log_dir))

  # Preserve original TiDE loop bound (epochs numbered 0 ... train_epochs).
  while int(epoch_counter.numpy()) < FLAGS.train_epochs + 1:
    ep = int(epoch_counter.numpy())
    logging.info('Epoch %s', ep)
    sys.stdout.flush()

    iterator = tqdm(dtl.tf_dataset(mode='train'), mininterval=2)
    for i, batch in enumerate(iterator):
      past_data = batch[:3]
      future_features = batch[4:6]
      tsidx = batch[-1]
      loss = model.train_step(
          past_data, future_features, batch[3], tsidx, optimizer
      )
      summary.update({'train/reg_loss': loss, 'train/loss': loss})
      if i % 100 == 0:
        mean_loss = summary.metric_dict['train/reg_loss'].result().numpy()
        iterator.set_description(f'Loss {mean_loss:.4f}')

    epoch_counter.assign_add(1)
    val_metrics, val_res, val_loss = model.eval(dtl, 'val', num_split=FLAGS.num_split)
    test_metrics, test_res, test_loss = model.eval(dtl, 'test', num_split=FLAGS.num_split)
    logging.info('Val Loss: %s, Test Loss: %s', val_loss, test_loss)

    tracked_loss = float(val_metrics['rmse'])
    early_stop = False
    if tracked_loss < float(best_rmse.numpy()) and ep > FLAGS.min_num_epochs:
      best_rmse.assign(tracked_loss)
      patience_counter.assign(0)
      _save_best_outputs(results_dir, val_res, test_res, test_metrics)
      logging.info('Saved best results to %s (val RMSE %.6f)', results_dir, tracked_loss)
      logging.info('Test metrics: %s', test_metrics)
      if backup:
        # Persist best outputs on each improvement; they must survive an interrupted session.
        backup.upload_results(results_dir=run_root / 'results')
    else:
      patience_counter.assign_add(1)
      if int(patience_counter.numpy()) > FLAGS.patience:
        logging.info('Early stopping')
        early_stop = True

    summary.write(step=int(epoch_counter.numpy()))
    end_of_training = int(epoch_counter.numpy()) >= FLAGS.train_epochs + 1
    finished.assign(early_stop or end_of_training)

    # Save the complete epoch state locally BEFORE uploading it to Drive.
    ckpt_path = manager.save(checkpoint_number=int(epoch_counter.numpy()))
    logging.info('Saved checkpoint at end of epoch %d: %s',
                 int(epoch_counter.numpy()), ckpt_path)
    if backup and (
        int(epoch_counter.numpy()) % FLAGS.backup_every == 0 or bool(finished.numpy())
    ):
      backup.backup_checkpoint(
          epoch=int(epoch_counter.numpy()), checkpoint_dir=ckpt_dir
      )

    if early_stop:
      break

  logging.info('Training ended at completed_epochs=%d, best val RMSE=%.6f',
               int(epoch_counter.numpy()), float(best_rmse.numpy()))


class Summary:
  """Summary statistics (unchanged from the provided TiDE implementation)."""

  def __init__(self, log_dir):
    self.metric_dict = {}
    self.writer = tf.summary.create_file_writer(log_dir)

  def update(self, update_dict):
    for metric in update_dict:
      if metric not in self.metric_dict:
        self.metric_dict[metric] = keras.metrics.Mean()
      self.metric_dict[metric].update_state(values=[update_dict[metric]])

  def write(self, step):
    with self.writer.as_default():
      for metric in self.metric_dict:
        tf.summary.scalar(metric, self.metric_dict[metric].result(), step=step)
    self.metric_dict = {}
    self.writer.flush()


def main(_):
  training()


if __name__ == '__main__':
  app.run(main)
