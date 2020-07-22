# Copyright (C) 2018 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from lottery_ticket.foundations import paths
from lottery_ticket.experiment import locations
import tensorflow as tf
import os


HYPERPARAMETERS = {'layers': [(256, tf.nn.relu), (256, tf.nn.relu), (10, None)], 
  'conv_kernel': 3, 
  'conv_stride': 2, 
  'conv2d': [(64, tf.nn.relu), (64, tf.nn.relu), (None, "maxpool")],
  'conv4d' : [(64, tf.nn.relu), (64, tf.nn.relu), (None, "maxpool"), (128, tf.nn.relu), (128, tf.nn.relu), (None, "maxpool")]}


DATASET_LOCATION = locations.DATASET_LOCATION

OPTIMIZER_FN = functools.partial(tf.compat.v1.train.AdamOptimizer, 2e-4)

PRUNE_PERCENTS = {'layer0': .2, 'layer1': .2, 'layer2': .1, 'conv2d0': .1, 'conv2d1': .1,
'conv4d0': .1, 'conv4d1': .1, 'conv4d2': .1, 'conv4d3': .1}


TRAINING_LEN = ('iterations', 50000)

EXPERIMENT_PATH = locations.EXPERIMENT_PATH


def graph(category, filename):
  return os.path.join(EXPERIMENT_PATH, 'graphs', category, filename)


def initialization(level):
  return os.path.join(EXPERIMENT_PATH, 'weights', str(level), 'initialization')


def trial(trial_name):
  return paths.trial(EXPERIMENT_PATH, trial_name)


def run(trial_name, level, experiment_name='same_init', run_id=''):
  return paths.run(trial(trial_name), level, experiment_name, run_id)
