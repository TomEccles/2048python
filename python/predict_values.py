import numpy as np
import tensorflow as tf

from file_utils import create_dir_if_needed

num_inputs = 16 + 16 + 16 * 16 + 4 + 1
num_outputs = 1
batch_size = 128
nodes_1 = 1024
# nodes_2 = 128
dropout = 0.5

value_graph = tf.Graph()

def calc(x, keep_param=1):
    mat_1 = tf.matmul(x, weights_1) + biases_1
    rel_1 = tf.nn.relu(mat_1)
    drop_1 = tf.nn.dropout(rel_1, keep_param)
    return tf.matmul(drop_1, weights_3) + biases_3


with value_graph.as_default():
    X = tf.placeholder(tf.float32, shape=(None, num_inputs))
    Y = tf.placeholder(tf.float32, shape=None)
    keep_param = tf.placeholder(tf.float32)

    # Variables.
    weights_1 = tf.Variable(
        tf.truncated_normal([num_inputs, nodes_1], stddev=0.01))
    biases_1 = tf.Variable(tf.zeros([nodes_1]))
    weights_3 = tf.Variable(
        tf.truncated_normal([nodes_1, 2], stddev=0.01))
    biases_3 = tf.Variable(tf.constant(1, dtype=tf.float32, shape=[2]))

    # Training computation.
    p, s = tf.split(calc(X, keep_param), [1, 1], 1)
    predictions = tf.reshape(p, [-1])
    std_devs = tf.reshape(s, [-1])
    mid = (predictions - Y) / std_devs
    total_loss = tf.nn.l2_loss(mid) + tf.reduce_sum(tf.log(std_devs))
    loss = total_loss / tf.cast(tf.shape(Y)[0], tf.float32)
    l2_loss = tf.nn.l2_loss(predictions - Y) / tf.cast(tf.shape(Y)[0], tf.float32)

    # Optimizer.
    optimizer = tf.train.AdamOptimizer(1e-6).minimize(loss)


class ValuerNet:
    def __init__(self):
        self.graph, self.session = self.setup_graph()

    def setup_graph(self):
        session = tf.Session(graph=value_graph)
        with value_graph.as_default():
            tf.initialize_all_variables().run(session=session)
        print("Initialized valuer")
        return value_graph, session

    def feed_observations(self, dataset, labels, passes):
        for p in range(passes):
            permuted_data, permuted_labels = self.permute(dataset, labels)
            for i in range(int(len(dataset) / batch_size)):
                batch_data = permuted_data[i * batch_size:(i + 1) * batch_size]
                batch_labels = permuted_labels[i * batch_size:(i + 1) * batch_size]
                feed_dict = {X: batch_data, Y: batch_labels, keep_param: 1 - dropout}
                _, l = self.session.run([optimizer, loss], feed_dict=feed_dict)

    def save(self, save_path):
        create_dir_if_needed(save_path)
        with self.graph.as_default():
            tf.train.Saver().save(self.session, save_path)

    def validate_observations(self, dataset, labels):
        p, l = self.session.run([predictions, l2_loss], {X: dataset, Y: labels, keep_param: 1})
        #pred, std, total_l, l, l2, m = self.session.run([predictions, std_devs, total_loss, loss, l2_loss, mid], {X: dataset, Y: labels, keep_param: 1})
        print("Validation loss, root mean square: %.4f %.4f" % (l, l ** .5))

    def run_forward(self, dataset):
        p = self.session.run([predictions], {X: dataset, keep_param: 1})
        return p[0]

    def load(self, save_path):
        with self.graph.as_default():
            tf.train.Saver().restore(self.session, save_path)

    def permute(self, dataset, labels):
        permutation = np.random.permutation(dataset.shape[0])
        shuffled_dataset = dataset[permutation, :]
        shuffled_labels = labels[permutation]
        return shuffled_dataset, shuffled_labels
