# lottery-ticket-hypothesis
Extension of the "lottery ticket hypothesis" project implemented by google-research



This is an extension of https://github.com/google-research/lottery-ticket-hypothesis, due to
their limitations on implementing CNN networks to work with them.

The new features are:


1) Added Convolution layer and MaxPooling layer, surrounding tf.Conv2d layer and tf.Maxpool, but with masks.

2) You can choose which network do you want to run: lenet, Conv2d and Conv4d (you can extending that)

3) In download_data.py you can choose which dataset do you want to download - mnist, fashion mnist and cifar10 *

4) A new UI for running an experiment for all models and dataset (in the original code, there was only mnist_fc experiment)



Getting Started

1) Download and install the original project https://github.com/google-research/lottery-ticket-hypothesis

2) Replace the current project files with the original files

3) Download data with experiment/download_data.py (mnist - mnist, fashion - fashion mnist, cifar10 - cifar10)

4) Run experiment/lottery_experiment.py

5) Get a test accuracy graph named "results.png"
