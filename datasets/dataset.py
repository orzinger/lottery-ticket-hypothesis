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


import keras
from lottery_ticket.foundations import dataset_base
from lottery_ticket.foundations import save_restore
import numpy as np


class Dataset(dataset_base.DatasetBase):

  def __init__(self,
               ds_location,
               permute_labels=False,
               train_order_seed=None):
    """Create an dataset object.

    Args:
      location: The directory that contains dataset as four npy files.
      permute_labels: Whether to randomly permute the labels.
      train_order_seed: (optional) The random seed for shuffling the training
        set.
    """
    dataset = save_restore.restore_network(ds_location)

    x_train = dataset['x_train']
    x_test = dataset['x_test']
    y_train = dataset['y_train']
    y_test = dataset['y_test']

    if permute_labels:
      # Reassign labels according to a random permutation of the labels.
      permutation = np.random.permutation(10)
      y_train = permutation[y_train]
      y_test = permutation[y_test]

    # Normalize x_train and x_test.
    x_train = keras.utils.normalize(x_train).astype(np.float32)
    x_test = keras.utils.normalize(x_test).astype(np.float32)

    # Convert y_train and y_test to one-hot.
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)


    x_val = x_train[x_train.shape[0] - 5000:]
    x_train = x_train[:x_train.shape[0] - 5000]
    y_val = y_train[y_train.shape[0] - 5000:]
    y_train = y_train[:y_train.shape[0] - 5000]

    # Prepare the dataset.
    super(Dataset, self).__init__(
        (x_train, y_train),
        64, (x_test, y_test),
        validate = (x_val, y_val),
        train_order_seed=train_order_seed)
