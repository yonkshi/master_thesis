# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import random
import os
import datetime
from scipy.misc import imsave

from model import VAE
from data_manager import DataManager

tf.app.flags.DEFINE_integer("epoch_size", 2000, "epoch size")
tf.app.flags.DEFINE_integer("batch_size", 64, "batch size")
tf.app.flags.DEFINE_float("gamma", 1000.0, "gamma param for latent loss")
tf.app.flags.DEFINE_float("capacity_limit", 30.0,
                          "encoding capacity limit param for latent loss")
tf.app.flags.DEFINE_integer("capacity_change_duration", 100000,
                            "encoding capacity change duration")
tf.app.flags.DEFINE_float("learning_rate", 5e-4, "learning rate")
tf.app.flags.DEFINE_string("checkpoint_dir", "checkpoints", "checkpoint directory")
tf.app.flags.DEFINE_string("log_file", "./log", "log file directory")
tf.app.flags.DEFINE_boolean("training", True, "training or not")

tf.app.flags.DEFINE_integer("input_width", 128, "input image pixel width")
tf.app.flags.DEFINE_integer("input_height", 128, "input image pixel height")
tf.app.flags.DEFINE_integer("input_channels", 3, "input image color channels (RGB default)")
tf.app.flags.DEFINE_integer("latent_dim", 256, "Dimension of latent space")

tf.app.flags.DEFINE_string("dataset", "Megaman", "log file directory")

flags = tf.app.flags.FLAGS

run_name = "gamma={0}, capacity_lim={1}, latent_dim={2}, input_dim={3}x{4}x{5}, dataset={6}, " \
           "date={7}".format(flags.gamma,
                              flags.capacity_limit,
                              flags.latent_dim,
                              flags.input_width,
                              flags.input_height,
                              flags.input_channels,
                              flags.dataset,
                              datetime.datetime.now(),
                              )
run_logpath = os.path.join(flags.log_file, run_name)
run_checkpoint_path = os.path.join(flags.checkpoint_dir, run_name)
if not os.path.exists(run_logpath):
    os.mkdir(run_logpath)

def train(sess,
          model,
          manager,
          saver):


    summary_writer = tf.summary.FileWriter(run_logpath, sess.graph)

    n_samples = manager.sample_size
    np.random.seed(1231)
    reconstruct_check_images = manager.get_random_images(10)

    indices = list(range(n_samples))

    step = 0

    # Training cycle
    for epoch in range(flags.epoch_size):
        print('\n===== EPOCH %d =====' % epoch)
        # Shuffle image indices
        random.shuffle(indices)

        avg_cost = 0.0
        total_batch = n_samples // flags.batch_size
        print('>> Total Batch Size: %d' % total_batch)

        # Loop over all batches
        print('>> Training ', end='')

        for i in range(total_batch):
            # Generate image batch
            print(".", end='')
            batch_indices = indices[flags.batch_size * i: flags.batch_size * (i + 1)]
            batch_xs = manager.get_images(batch_indices)

            # Fit training using batch data
            reconstr_loss, latent_loss, summary_str = model.partial_fit(sess, batch_xs, step)
            summary_writer.add_summary(summary_str, step)
            step += 1

        # Image reconstruction check
        print('')
        print('>> Reconstruction check ... ', end='')
        img_summary = reconstruct_check(sess, model, reconstruct_check_images)
        print('Done')

        # # Disentangle check
        # print('>> Disentanglement check...', end='')
        # disentangle_check(sess, model, manager)
        # print('Done')

        summary_writer.add_summary(img_summary, step)

        # Save checkpoint
        saver.save(sess, run_checkpoint_path + '/' + 'checkpoint', global_step=step)


def reconstruct_check(sess, model, images):
    # Check image reconstruction

    x_reconstruct, img_summary = model.reconstruct(sess, images)

    if not os.path.exists("reconstr_img"):
        os.mkdir("reconstr_img")

    for i in range(len(images)):
        print('>>>> Reconstructing image %d ' % i)
        org_img = images[i].reshape([flags.input_width, flags.input_height, flags.input_channels])
        org_img = org_img.astype(np.float32)
        reconstr_img = x_reconstruct[i].reshape([flags.input_width, flags.input_height, flags.input_channels])
        imsave("reconstr_img/org_{0}.png".format(i), org_img)
        imsave("reconstr_img/reconstr_{0}.png".format(i), reconstr_img)

    return img_summary


def disentangle_check(sess, model, manager, save_original=False):
    '''
    This code appears to be running disentanglement check (specified in the paper) with a preselected image.
    So in my case, I am running with the preselected image 1337
    :param sess:
    :param model:
    :param manager:
    :param save_original:
    :return:
    '''
    img = manager.get_image(1337)
    if save_original:
        imsave("original.png",
               img.reshape([flags.input_width, flags.input_height, flags.input_channels]).astype(np.float32))

    batch_xs = [img]
    z_mean, z_log_sigma_sq = model.transform(sess, batch_xs)
    z_sigma_sq = np.exp(z_log_sigma_sq)[0]

    # Print variance
    zss_str = ""
    for i, zss in enumerate(z_sigma_sq):
        str = "z{0}={1:.4f}".format(i, zss)
        zss_str += str + ", "
    # print(zss_str)

    # Save disentangled images
    z_m = z_mean[0]
    n_z = 256  # Latent space dim

    if not os.path.exists("disentangle_img"):
        os.mkdir("disentangle_img")

    for target_z_index in range(n_z):
        for ri in range(n_z):
            value = -3.0 + (6.0 / 9.0) * ri
            z_mean2 = np.zeros((1, n_z))
            for i in range(n_z):
                if (i == target_z_index):
                    z_mean2[0][i] = value
                else:
                    z_mean2[0][i] = z_m[i]
            reconstr_img = model.generate(sess, z_mean2)
            rimg = reconstr_img[0].reshape([flags.input_width, flags.input_height, flags.input_channels])
            imsave("disentangle_img/check_z{0}_{1}.png".format(target_z_index, ri), rimg)


def load_checkpoints(sess):
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state(run_checkpoint_path)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("loaded checkpoint: {0}".format(checkpoint.model_checkpoint_path))
    else:
        print("Could not find old checkpoint")
        if not os.path.exists(run_checkpoint_path):
            os.mkdir(run_checkpoint_path)
    return saver


def main(argv):
    manager = DataManager()
    manager.load()

    sess = tf.Session()

    model = VAE(
        input_width=flags.input_width,
        input_height=flags.input_height,
        input_channels=flags.input_channels,
        gamma=flags.gamma,
        capacity_limit=flags.capacity_limit,
        capacity_change_duration=flags.capacity_change_duration,
        learning_rate=flags.learning_rate,
    )

    sess.run(tf.global_variables_initializer())

    saver = load_checkpoints(sess)

    if flags.training:
        # Train
        train(sess, model, manager, saver)
    else:
        reconstruct_check_images = manager.get_random_images(10)
        # Image reconstruction check
        reconstruct_check(sess, model, reconstruct_check_images)
        # Disentangle check
        disentangle_check(sess, model, manager)


if __name__ == '__main__':
    tf.app.run()
