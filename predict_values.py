import numpy as np
import tensorflow as tf

from file_utils import create_dir_if_needed

num_inputs = 16 + 16 + 16 * 16 + 4 + 1
num_outputs = 1
batch_size = 128
nodes_1 = 512
# nodes_2 = 128
dropout = 0.5


def accuracy_matrix(pred, labels):
    size = 4
    for i in range(size):
        for j in range(size):
            total = sum([a[i] for a in labels])

            matches = sum([a[j] * b[i] for a, b in zip(pred, labels)])
            print("%.3f" % (matches / total), end=",")
        print()


def permute(dataset, labels):
    permutation = np.random.permutation(dataset.shape[0])
    shuffled_dataset = dataset[permutation, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels


entropy_graph = tf.Graph()


def calc(x, keep_param=1):
    mat_1 = tf.matmul(x, weights_1) + biases_1
    rel_1 = tf.nn.relu(mat_1)
    drop_1 = tf.nn.dropout(rel_1, keep_param)
    return tf.matmul(drop_1, weights_3) + biases_3


with entropy_graph.as_default():
    X = tf.placeholder(tf.float32, shape=(None, num_inputs))
    Y = tf.placeholder(tf.float32, shape=(None))
    keep_param = tf.placeholder(tf.float32)

    # Variables.
    weights_1 = tf.Variable(
        tf.truncated_normal([num_inputs, nodes_1], stddev=0.01))
    biases_1 = tf.Variable(tf.zeros([nodes_1]))
    weights_3 = tf.Variable(
        tf.truncated_normal([nodes_1, 1], stddev=0.01))
    biases_3 = tf.Variable(tf.zeros([1]))

    # Training computation.
    predictions = tf.reshape(calc(X, keep_param),[-1])
    loss = tf.reduce_mean(tf.nn.l2_loss(predictions - Y)) / tf.cast(tf.shape(Y)[0], tf.float32)

    # Optimizer.
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)


class ValuerNet:
    def __init__(self):
        self.session = tf.Session(graph=entropy_graph)
        with entropy_graph.as_default():
            tf.initialize_all_variables().run(session=self.session)
        print("Initialized valuer")

    def feed_observations(self, dataset, labels, passes):
        for p in range(passes):
            permuted_data, permuted_labels = permute(dataset, labels)
            for i in range(int(len(dataset) / batch_size)):
                batch_data = permuted_data[i * batch_size:(i + 1) * batch_size]
                batch_labels = permuted_labels[i * batch_size:(i + 1) * batch_size]
                feed_dict = {X: batch_data, Y: batch_labels, keep_param: 1 - dropout}
                _, l = self.session.run([optimizer, loss], feed_dict=feed_dict)

    def save(self, save_path):
        create_dir_if_needed(save_path)
        with entropy_graph.as_default():
            tf.train.Saver().save(self.session, save_path)

    def validate_observations(self, dataset, labels):
        p, l = self.session.run([predictions, loss], {X: dataset, Y: labels, keep_param: 1})
        print("Validation loss, root mean square: %.4f %.4f" % (l, l**.5))


    def run_forward(self, data_point):
        p = self.session.run([predictions], {X: [data_point]})
        return p[0][0]

    def load(self, save_path):
        with entropy_graph.as_default():
            tf.train.Saver().restore(self.session, save_path)
