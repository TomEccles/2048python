import numpy as np
import tensorflow as tf

from file_utils import create_dir_if_needed

num_inputs = 16 + 16 + 16 * 16
num_outputs = 4
batch_size = 128
nodes_1 = 512
# nodes_2 = 128
dropout = 0.5


def accuracy_matrix(predictions, labels):
    size = 4
    for i in range(size):
        for j in range(size):
            total = sum([a[i] for a in labels])

            matches = sum([a[j] * b[i] for a, b in zip(predictions, labels)])
            print("%.3f" % (matches / total), end=",")
        print()


def permute(dataset, labels):
    permutation = np.random.permutation(dataset.shape[0])
    shuffled_dataset = dataset[permutation, :]
    shuffled_labels = labels[permutation, :]
    return shuffled_dataset, shuffled_labels


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


prior_graph = tf.Graph()


def calc(x, keep_param=1):
    mat_1 = tf.matmul(x, weights_1) + biases_1
    rel_1 = tf.nn.relu(mat_1)
    drop_1 = tf.nn.dropout(rel_1, keep_param)
    # mat_2 = tf.matmul(drop_1, weights_2) + biases_2
    # rel_2 = tf.nn.relu(mat_2)
    # drop_2 = tf.nn.dropout(rel_2, keep_param)
    return tf.matmul(drop_1, weights_3) + biases_3


with prior_graph.as_default():
    X = tf.placeholder(tf.float32, shape=(None, num_inputs))
    Y = tf.placeholder(tf.float32, shape=(None, num_outputs))
    g = tf.placeholder(tf.float32, shape=(None, num_outputs))
    keep_param = tf.placeholder(tf.float32)

    # Variables.
    weights_1 = tf.Variable(
        tf.truncated_normal([num_inputs, nodes_1], stddev=0.01))
    biases_1 = tf.Variable(tf.zeros([nodes_1]))
    # weights_2 = tf.Variable(
    #    tf.truncated_normal([nodes_1, nodes_2], stddev=0.001))
    # biases_2 = tf.Variable(tf.zeros([nodes_2]))
    weights_3 = tf.Variable(
        tf.truncated_normal([nodes_1, num_outputs], stddev=0.01))
    biases_3 = tf.Variable(tf.zeros([num_outputs]))

    # Training computation.
    logits = calc(X, keep_param)
    unnorm_pred = tf.nn.softmax(logits) * g
    pred = unnorm_pred / tf.reshape(tf.reduce_sum(unnorm_pred, 1), (-1, 1))
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y))

    # Optimizer.
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)


class PriorNet(object):
    def __init__(self):
        self.session = tf.Session(graph=prior_graph)
        with prior_graph.as_default():
            tf.initialize_all_variables().run(session=self.session)
        print("Initialized prior net")

    def feed_observations(self, dataset, labels, passes):
        for p in range(passes):
            permuted_data, permuted_labels = permute(dataset, labels)
            gates = np.array([a[-4:] for a in permuted_data])
            permuted_data = permuted_data[:, :-4]
            for i in range(int(len(dataset) / batch_size)):
                batch_data = permuted_data[i * batch_size:(i + 1) * batch_size]
                batch_labels = permuted_labels[i * batch_size:(i + 1) * batch_size]
                batch_gates = gates[i * batch_size:(i + 1) * batch_size]
                feed_dict = {X: batch_data, Y: batch_labels, g: batch_gates, keep_param: 1 - dropout}
                _, l, p = self.session.run([optimizer, loss, pred], feed_dict=feed_dict)

    def save(self, save_path):
        create_dir_if_needed(save_path)
        with prior_graph.as_default():
            tf.train.Saver().save(self.session, save_path)

    def validate_observations(self, dataset, labels):
        gates = np.array([a[-4:] for a in dataset])
        dataset = dataset[:, :-4]
        p, l = self.session.run([pred, loss], {X: dataset, Y: labels, g: gates, keep_param: 1})
        a = accuracy(p, labels)
        print("Validation accuracy: %.1f%%" % a)
        print("Validation loss: %.4f" % l)
        accuracy_matrix(p, labels)

    def run_forward(self, data_point):
        gates = [data_point[-4:]]
        dataset = [data_point[:-4]]
        p = self.session.run([pred], {X: dataset, g: gates, keep_param: 1})
        return p[0][0]

    def load(self, save_path):
        with prior_graph.as_default():
            tf.train.Saver().restore(self.session, save_path)
