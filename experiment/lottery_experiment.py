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

"""The lottery ticket experiment for all architactures"""

# from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from lottery_ticket.datasets import dataset
from lottery_ticket.foundations import experiment
from lottery_ticket.foundations import model
from lottery_ticket.foundations import paths
from lottery_ticket.foundations import pruning
from lottery_ticket.foundations import save_restore
from lottery_ticket.foundations import trainer
from lottery_ticket.experiment import constants
import matplotlib.pyplot as plt
import argparse
import numpy as np



def train(output_dir,
          iterations,
          conv_layers,
          experiment_name,
          training_len=constants.TRAINING_LEN,
          location=constants.DATASET_LOCATION,
          presets=None,
          permute_labels=False,
          train_order_seed=None):
  """Perform the lottery ticket experiment.

  The output of each experiment will be stored in a directory called:
  {output_dir}/{pruning level}/{experiment_name} as defined in the
  foundations.paths module.

  Args:
    output_dir: Parent directory for all output files.
    location: The path to the NPZ file containing dataset.
    training_len: How long to train on each iteration.
    iterations: How many iterative pruning steps to perform.
    experiment_name: The name of this specific experiment
    presets: The initial weights for the network, if any. Presets can come in
      one of three forms:
      * A dictionary of numpy arrays. Each dictionary key is the name of the
        corresponding tensor that is to be initialized. Each value is a numpy
        array containing the initializations.
      * The string name of a directory containing one file for each
        set of weights that is to be initialized (in the form of
        foundations.save_restore).
      * None, meaning the network should be randomly initialized.
    permute_labels: Whether to permute the labels on the dataset.
    train_order_seed: The random seed, if any, to be used to determine the
      order in which training examples are shuffled before being presented
      to the network.
  """
  # Define model and dataset functions.
  def make_dataset():
    return dataset.Dataset(
        location,
        permute_labels=permute_labels,
        train_order_seed=train_order_seed)

  make_model = functools.partial(model.Model, constants.HYPERPARAMETERS)

  # Define a training function.
  def train_model(_sess, _level, _dataset, _model):
    params = {
        'test_interval': 100,
        'save_summaries': True,
        'save_network': True,
    }
  
    return trainer.train(
        _sess,
        _dataset,
        _model,
        constants.OPTIMIZER_FN,
        training_len,
        output_dir=paths.run(output_dir, _level, experiment_name),
        **params)

  # Define a pruning function.
  prune_masks = functools.partial(pruning.prune_by_percent,
                                  constants.PRUNE_PERCENTS)

  # Run the experiment
  t_accuracy, v_loss = experiment.experiment(
      make_dataset,
      make_model,
      train_model,
      prune_masks,
      iterations,
      conv_layers,
      presets=save_restore.standardize(presets))

  for k, x in t_accuracy.items():

    plt.plot(np.arange(0, constants.TRAINING_LEN[1], constants.TRAINING_LEN[1]%100), x, linewidth = 0.8, label = k)

  plt.legend()

  plt.xticks(np.arange(0, constants.TRAINING_LEN[1], 500))

  plt.grid()

  plt.savefig("experminets_graphs/results.png")


if __name__ == "__main__":
  
  parser = argparse.ArgumentParser()
  parser.add_argument("outputdir", type = str, help = "output directory")
  parser.add_argument("-i", "--iterations", type = int, help = "pruning iterations", default = 2)
  parser.add_argument("-c", "--convlayers", type = int, help = "how many conv layers", default = 2)
  parser.add_argument("-e", "--experiment_name", type = str, help = "experiment name", default = "same_init")
  parser.add_argument("-t", "--trainings", type = int, help = "trainings iterations", default = 500)
  args = parser.parse_args()
  train(args.outputdir, iterations = args.iterations, training_len = ("iterations", args.trainings), conv_layers = args.convlayers, experiment_name = args.experiment_name)
  
    
