import tensorflow as tf
import math
import numpy as np
import os
from six.moves import xrange
from tensorflow.contrib.tensorboard.plugins import projector
import matplotlib.pyplot as plt
from tempfile import gettempdir
import DataPretreat

class SkipGram:
    def __init__(self, vocabulary_size,
                       batch_size=128,
                       embedding_size=128,
                       num_sampled=64,
                       valid_window=100,
                       valid_size=16,
                       log_dir="/Users/cly/PycharmProjects/Word2Vec/log/"):
        self.vocabulary_size = vocabulary_size
        self.batch_size      = batch_size
        self.embedding_size  = embedding_size
        self.num_sampled     = num_sampled
        self.valid_size      = valid_size
        self.valid_example   = np.random.choice(valid_window, valid_size, replace=False)
        self.log_dir         = log_dir

    def build_network(self):
        self.graph = tf.Graph()

        with self.graph.as_default():
            with tf.name_scope('inputs'):
                # 首先定义两个用作输入的占位符，分别输入输入集(train_inputs)和标签集(train_labels)
                self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
                self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
                # 验证集
                self.valid_dataset = tf.constant(self.valid_example, dtype=tf.int32)

            with tf.device('/cpu:0'):
                with tf.name_scope('embeddings'):
                    # 词向量矩阵，初始时为均匀随机正态分布
                    self.embeddings = tf.Variable(
                        tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0)
                    )
                    # 将输入序列向量化
                    self.embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)

                # Construct the variables for the NCE loss
                with tf.name_scope('weights'):
                    self.nce_weights = tf.Variable(
                        tf.truncated_normal(
                            [self.vocabulary_size, self.embedding_size], stddev=1.0/math.sqrt(self.embedding_size)
                        )
                    )

                with tf.name_scope('biases'):
                    self.nce_biases = tf.Variable(tf.zeros([self.vocabulary_size]))

            # Compute the average NCE loss for the batch.
            with tf.name_scope('loss'):
                self.loss = tf.reduce_mean(
                    tf.nn.nce_loss(weights=self.nce_weights,         # 权重
                                   biases= self.nce_biases,          # 偏差
                                   labels=self.train_labels,         # 输入的标签
                                   inputs=self.embed,                # 输入向量
                                   num_sampled=self.num_sampled,     # 负采样的个数
                                   num_classes=self.vocabulary_size) # 类别数目
                )

                # 与tensorboard 相关，让tensorflow记录loss参数
                tf.summary.scalar('loss', self.loss)

            with tf.name_scope('optimizer'):
                # 根据 nce loss 来更新梯度和embedding，使用梯度下降法(gradient descent)来实现
                self.optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(self.loss)

            # Compute the cosine similarity between minibatch examples and all embeddings
            norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keepdims=True))
            # 对embedding向量正则化
            self.normalized_embeddings = self.embeddings / norm
            valid_embeddings = tf.nn.embedding_lookup(self.normalized_embeddings, self.valid_dataset)
            self.similarity = tf.matmul(valid_embeddings, self.normalized_embeddings, transpose_b=True)

            # 汇总所有的变量记录
            self.merged = tf.summary.merge_all()

            # 变量初始化操作
            self.init = tf.global_variables_initializer()

            # 保存模型的操作
            self.saver = tf.train.Saver()



    def train(self, num_steps=100001):
        with tf.Session(graph=self.graph) as session:
            self.writer = tf.summary.FileWriter(self.log_dir, session.graph)

            # We must initialize all variables before we use them.
            self.init.run()
            print("Initialized")

            average_loss = 0
            for step in xrange(num_steps):
                batch_inputs, batch_labels = DataPretreat.generate_batch(self.batch_size)
                feed_dict = {self.train_inputs:batch_inputs, self.train_labels:batch_labels}

                # Define metadata variable.
                self.run_metadata = tf.RunMetadata()

                _, summary, loss_val = session.run(
                    [self.optimizer, self.merged, self.loss], feed_dict=feed_dict, run_metadata=self.run_metadata
                )
                average_loss += loss_val

                # Add returned summaries to writer in each step.
                self.writer.add_summary(summary, step)

                # Add metadata to visualize the graph for the last run.
                if step == (num_steps - 1):
                    self.writer.add_run_metadata(self.run_metadata, 'step%d' %step)

                if step % 2000 == 0:
                    if step > 0:
                        average_loss /= 2000
                    print('Average loss at step ', step, ": ", average_loss)
                    average_loss = 0

                # Note that this is expensive (~20% slowdown if computed every 500 steps)
                if step % 10000 == 0:
                    sim = self.similarity.eval()
                    for i in xrange(self.valid_size):
                        valid_word = DataPretreat.index_to_word_map[self.valid_example[i]]
                        top_k = 8
                        nearest = (-sim[i,:]).argsort()[1:top_k+1]
                        log_str = "Nearest to %s:" % valid_word
                        for k in xrange(top_k):
                            close_word = DataPretreat.index_to_word_map[nearest[k]]
                            log_str = "%s %s," % (log_str, close_word)
                        print(log_str)

            self.final_embeddings = self.normalized_embeddings.eval()

            with open(self.log_dir + "/metadata.tsv","w") as f:
                for i in xrange(self.vocabulary_size):
                    f.write(DataPretreat.index_to_word_map[i] + "\n")

                # Save the model for checkpoints.
                self.saver.save(session, os.path.join(self.log_dir, "model.ckpt"))

                # Create a configuration for visualizing embeddings with the labels in TensorBoard.
                config = projector.ProjectorConfig()
                embedding_conf = config.embeddings.add()
                embedding_conf.tensor_name = self.embeddings.name
                # Specify where you find the metadata
                embedding_conf.metadata_path = os.path.join(self.log_dir, "metadata.tsv")
                # Specify where you find the sprite (we will create this later)
                # embedding_conf.sprite.image_path = os.path.join(self.log_dir, "sprite.png")
                # embedding_conf.sprote.single_image_dim.extend([28,28])
                # Say that you want to visualise the embeddings
                projector.visualize_embeddings(self.writer, config)


            self.writer.close()

    # draw visualization of distance between embeddings.
    def visualize(self):
        try:
            # pylint: disable=g-import-not-at-top
            from sklearn.manifold import TSNE
            import matplotlib.pyplot as plt

            #为了在图片上能显示出中文
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

            tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
            plot_only = 500
            low_dim_embs = tsne.fit_transform(self.final_embeddings[:plot_only, :])
            labels = [DataPretreat.index_to_word_map[i] for i in xrange(plot_only)]
            self.__plot_with_labels(low_dim_embs, labels, os.path.join(self.log_dir, 'tsne.png'))
        except ImportError as ex:
            print('Please install sklearn, matplotlib, and scipy to show embeddings.')
            print(ex)

    def __plot_with_labels(self, low_dim_embs, labels, filename, fonts=None):
        assert low_dim_embs.shape[0] >= len(labels) , 'More labels than embeddings'
        plt.figure(figsize=(18,18))
        for i, label in enumerate(labels):
            x, y = low_dim_embs[i,:]
            plt.scatter(x, y)
            plt.annotate(label,
                         fontproperties=fonts,
                         xy=(x, y),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')

        plt.savefig(filename, dpi=600)

if __name__ == "__main__":
    print("数据预处理阶段")
    DataPretreat.prepare_data(windows_size=3)
    vocabulary_size = len(DataPretreat.vocabulary_list)
    Obj = SkipGram(vocabulary_size)
    print("创建SkipGram神经网络")
    Obj.build_network()
    print("训练SkipGram神经网络")
    Obj.train()
    print("可视化SkipGram训练效果")
    Obj.visualize()

