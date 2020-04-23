
import numpy as np
import tensorflow as tf

class EGES_Model:
    def __init__(self, num_nodes, feature_lens, side_info, batch_index, batch_labels, num_feat, n_sampled=100,
                 embedding_dim=128, lr=0.001):
        self.num_nodes = num_nodes
        self.feature_lens = feature_lens
        self.n_samped = n_sampled
        self.batch_index = batch_index
        self.batch_labels = tf.reshape(batch_labels, [-1, 1])
        self.side_info = tf.convert_to_tensor(side_info)
        self.batch_features = tf.nn.embedding_lookup(self.side_info, batch_index)
        self.num_feat = num_feat
        self.embedding_dim = embedding_dim
        self.num_samples = len(side_info)
        self.lr = lr
        self.softmax_w = tf.Variable(tf.truncated_normal((self.num_samples, embedding_dim), stddev=0.1), name='softmax_w')
        self.softmax_b = tf.Variable(tf.zeros(self.num_samples), name='softmax_b')
        self.cat_embedding = self.embedding_init()
        self.alpha_embedding = tf.Variable(tf.random_uniform((self.num_samples, num_feat), -1, 1))
        self.merge_emb = self.attention_merge()
        self.cost = self.make_skipgram_loss()
        # self.train_op = tf.train.AdagradOptimizer(lr).minimize(self.cost)
        self.train_op = tf.train.AdamOptimizer(lr).minimize(self.cost)

    def embedding_init(self):
        cat_embedding_vars = []
        for i in range(self.num_feat):
            embedding_var = tf.Variable(tf.random_uniform((self.feature_lens[i], self.embedding_dim), -1, 1), name='embedding'+str(i),
                                        trainable=True)
            cat_embedding_vars.append(embedding_var)
        return cat_embedding_vars

    def attention_merge(self):
        embed_list = []
        for i in range(self.num_feat):
            cat_embed = tf.nn.embedding_lookup(self.cat_embedding[i], self.batch_features[:, i])
            embed_list.append(cat_embed)
        stack_embed = tf.stack(embed_list, axis=-1)
        # attention merge
        alpha_embed = tf.nn.embedding_lookup(self.alpha_embedding, self.batch_features[:, 0])
        alpha_embed_expand = tf.expand_dims(alpha_embed, 1)
        alpha_i_sum = tf.reduce_sum(tf.exp(alpha_embed_expand), axis=-1)
        merge_emb = tf.reduce_sum(stack_embed * tf.exp(alpha_embed_expand), axis=-1) / alpha_i_sum
        return merge_emb

    def make_skipgram_loss(self):
        loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(
            weights=self.softmax_w,
            biases=self.softmax_b,
            labels=self.inputs[-1],
            inputs=self.merge_emb,
            num_sampled=self.n_samped,
            num_classes=self.num_nodes))
        return loss
