
import pandas as pd
import numpy as np
import tensorflow as tf
import time
import argparse
from EGES_model_dataset import EGES_Model
from utils import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--n_sampled", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--root_path", type=str, default='./data_cache/')
    parser.add_argument("--num_feat", type=int, default=4)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--outputEmbedFile", type=str, default='./embedding/EGES.embed')
    args = parser.parse_args()

    # read train_data
    print('read features...')
    start_time = time.time()
    side_info = np.loadtxt(args.root_path + 'sku_side_info.csv', dtype=np.int32, delimiter='\t')
    feature_lens = []
    for i in range(side_info.shape[1]):
        tmp_len = len(set(side_info[:, i]))
        feature_lens.append(tmp_len)
    end_time = time.time()
    print('time consumed for read features: %.2f' % (end_time - start_time))

    # read data_pair by tf.dataset
    def decode_data_pair(line):
        columns = tf.string_split([line], ' ')
        x = tf.string_to_number(columns.values[0], out_type=tf.int32)
        y = tf.string_to_number(columns.values[1], out_type=tf.int32)
        return x, y
    dataset = tf.data.TextLineDataset(args.root_path + 'all_pairs').map(decode_data_pair, num_parallel_calls=10).prefetch(500000)
    # dataset = dataset.shuffle(256)
    dataset = dataset.repeat(args.epochs)
    dataset = dataset.batch(args.batch_size)  # Batch size to use
    iterator = dataset.make_one_shot_iterator()
    batch_index, batch_labels = iterator.get_next()

    print('read embedding...')
    start_time = time.time()
    EGES = EGES_Model(len(side_info), feature_lens, side_info, batch_index, batch_labels, args.num_feat,
                      n_sampled=args.n_sampled, embedding_dim=args.embedding_dim, lr=args.lr)
    end_time = time.time()
    print('time consumed for read embedding: %.2f' % (end_time - start_time))

    # init model
    print('init...')
    start_time = time.time()
    init = tf.global_variables_initializer()
    config_tf = tf.ConfigProto()
    config_tf.gpu_options.allow_growth = True
    sess = tf.Session(config=config_tf)
    sess.run(init)
    end_time = time.time()
    print('time consumed for init: %.2f' % (end_time - start_time))

    print_every_k_iterations = 100
    loss = 0
    iteration = 0
    start = time.time()

    while True:
        try:
            iteration += 1
            _, train_loss = sess.run([EGES.train_op, EGES.cost])
            loss += train_loss

            if iteration % print_every_k_iterations == 0:
                end = time.time()
                print("Iteration: {}".format(iteration),
                      "Avg. Training loss: {:.4f}".format(loss / print_every_k_iterations),
                      "{:.4f} sec/batch".format((end - start) / print_every_k_iterations))
                loss = 0
                start = time.time()
        except tf.errors.OutOfRangeError as e:
            print(e)
            break

    print('optimization finished...')
    saver = tf.train.Saver()
    saver.save(sess, "model/EGES_cb")


    # feed_dict_test = {input_col: side_info[:, i] for i, input_col in enumerate(EGES.inputs[:-1])}
    # feed_dict_test[EGES.inputs[-1]] = np.zeros((len(side_info), 1), dtype=np.int32)

    feed_dict = {EGES.batch_index: side_info[:, 0]}
    embedding_result = sess.run(EGES.merge_emb, feed_dict=feed_dict)
    print('saving embedding result...')
    write_embedding(embedding_result, args.outputEmbedFile)

    print('visualization...')
    plot_embeddings(embedding_result[:5000, :], side_info[:5000, :])
