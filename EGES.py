
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from evaluate import plot_embeddings


def get_target(words, idx, window_size=5):
    R = np.random.randint(1, window_size + 1)
    start = idx - R if (idx - R) > 0 else 0
    stop = idx + R
    target_words = set(words[start:idx] + words[idx + 1:stop + 1])
    return list(target_words)


def get_batches(words, batch_size, window_size=5):
    ''' Create a generator of word batches as a tuple (inputs, targets) '''

    n_batches = len(words) // batch_size

    # only full batches
    words = words[:n_batches * batch_size]

    for idx in range(0, len(words), batch_size):
        x, y = [], []
        batch = words[idx:idx + batch_size]
        for ii in range(len(batch)):
            batch_x = batch[ii]
            batch_y = get_target(batch, ii, window_size)
            y.extend(batch_y)
            x.extend([batch_x] * len(batch_y))
        yield x, y


if __name__  == '__main__':

    with open('./data_cache/session_reproduce', 'rb') as f:
        session_list = pickle.load(f)
    # with open('./data_cache/side_info_list', 'rb') as f:
    #     side_infos = pickle.load(f)
    with open('./data_cache/side_info', 'rb') as f:
        side_info_dict = pickle.load(f)

    # brand_list, shop_list, cate_list = side_infos
    session_items = []
    for lst in session_list:
        lst = list(map(int, lst)) + [0]
        session_items.extend(lst)

    n_skus = len(side_info_dict.keys())
    n_brand = len(set(map(lambda x: x[0], side_info_dict.values())))
    n_shop = len(set(map(lambda x: x[1], side_info_dict.values())))
    n_cate = len(set(map(lambda x: x[2], side_info_dict.values())))
    n_sampled = 100
    embedding_dim = 128
    n_side_info = 4

    train_graph = tf.Graph()
    with train_graph.as_default():
        # define placeholder
        inputs_sku = tf.placeholder(tf.int32, [None], name='inputs_sku')
        inputs_brand = tf.placeholder(tf.int32, [None], name='inputs_brand')
        inputs_shop = tf.placeholder(tf.int32, [None], name='inputs_shop')
        inputs_cate = tf.placeholder(tf.int32, [None], name='inputs_cate')
        labels = tf.placeholder(tf.int32, [None, None], name='labels')

        # define embeddings
        sku_embedding = tf.Variable(tf.random_uniform((n_skus, embedding_dim), -1, 1))
        sku_embed = tf.nn.embedding_lookup(sku_embedding, inputs_sku)
        brand_embedding = tf.Variable(tf.random_uniform((n_brand, embedding_dim), -1, 1))
        brand_embed = tf.nn.embedding_lookup(brand_embedding, inputs_brand)
        shop_embedding = tf.Variable(tf.random_uniform((n_shop, embedding_dim), -1, 1))
        shop_embed = tf.nn.embedding_lookup(shop_embedding, inputs_shop)
        cate_embedding = tf.Variable(tf.random_uniform((n_cate, embedding_dim), -1, 1))
        cate_embed = tf.nn.embedding_lookup(cate_embedding, inputs_cate)
        concat_embed = tf.stack([sku_embed, brand_embed, shop_embed, cate_embed], axis=-1)
        # attention merge
        alpha_embedding = tf.Variable(tf.random_uniform((n_skus, n_side_info), -1, 1))
        alpha_embed = tf.nn.embedding_lookup(alpha_embedding, inputs_sku)
        alpha_embed_expand = tf.expand_dims(alpha_embed, 1)
        alpha_i_sum = tf.reduce_sum(tf.exp(alpha_embed_expand), axis=-1)
        merge_emb = tf.reduce_sum(concat_embed*tf.exp(alpha_embed_expand), axis=-1)/alpha_i_sum

        # compute loss
        softmax_w = tf.Variable(tf.truncated_normal((n_skus, embedding_dim), stddev=0.1))
        softmax_b = tf.Variable(tf.zeros(n_skus))

        # Calculate the loss using negative sampling
        loss = tf.nn.sampled_softmax_loss(softmax_w, softmax_b,
                                          labels, merge_emb,
                                          n_sampled, n_skus)

        cost = tf.reduce_mean(loss)
        optimizer = tf.train.AdamOptimizer().minimize(cost)

    # start train
    epochs = 10
    batch_size = 1024
    window_size = 5

    with train_graph.as_default():
        saver = tf.train.Saver()

    with tf.Session(graph=train_graph) as sess:
        iteration = 1
        loss = 0
        sess.run(tf.global_variables_initializer())

        for e in range(1, epochs + 1):
            batches = get_batches(session_items, batch_size, window_size)
            start = time.time()
            for sku_list, y in batches:
                brand_list = list(map(lambda x: side_info_dict[x][0], sku_list))
                shop_list = list(map(lambda x: side_info_dict[x][1], sku_list))
                cate_list = list(map(lambda x: side_info_dict[x][2], sku_list))
                feed = {inputs_sku: sku_list, inputs_brand: brand_list, inputs_shop: shop_list,
                        inputs_cate: cate_list, labels: np.array(y)[:, None]}
                train_loss, _ = sess.run([cost, optimizer], feed_dict=feed)

                loss += train_loss

                if iteration % 500 == 0:
                    end = time.time()
                    print("Epoch {}/{}".format(e, epochs),
                          "Iteration: {}".format(iteration),
                          "Avg. Training loss: {:.4f}".format(loss / 100),
                          "{:.4f} sec/batch".format((end - start) / 100))
                    loss = 0
                    start = time.time()

                iteration += 1
        save_path = saver.save(sess, "checkpoints/EGES.ckpt")

    # vis evaluate
    with tf.Session(graph=train_graph) as sess:
        saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
        batches = get_batches(session_items, 500, window_size)
        sku_list, y = next(batches)
        brand_list = list(map(lambda x: side_info_dict[x][0], sku_list))
        shop_list = list(map(lambda x: side_info_dict[x][1], sku_list))
        cate_list = list(map(lambda x: side_info_dict[x][2], sku_list))
        feed = {inputs_sku: sku_list, inputs_brand: brand_list, inputs_shop: shop_list,
                            inputs_cate: cate_list, labels: np.array(y)[:, None]}
        embed_mat = sess.run(merge_emb, feed_dict=feed)
        plot_embeddings(embed_mat, brand_list, shop_list, cate_list)

        # alpha_mat = sess.run(tf.exp(alpha_embedding))
