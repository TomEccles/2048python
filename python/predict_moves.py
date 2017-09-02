import numpy as np
import tensorflow as tf

from file_utils import create_dir_if_needed

num_inputs = 16 + 16 + 16 * 16 + 4 + 1
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


def permute(*arrays):
    permutation = np.random.permutation(arrays[0].shape[0])
    return [a[permutation, :] for a in arrays]


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
    V = tf.placeholder(tf.float32, shape=(None))
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

    # Shared computation
    logits = calc(X, keep_param)

    # Run forward computation
    probs = tf.nn.softmax(logits)

    # Supervised training computation.
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(1e-5).minimize(loss)

    # Reinforcement computation
    a = tf.reduce_sum(probs * Y, axis=1)
    probs_for_actions = tf.log(a)
    value = tf.reduce_mean(probs_for_actions * V)
    reinforcement_optimizer = tf.train.AdamOptimizer(1e-7).minimize(-value)



class PriorNet(object):
    def __init__(self):
        self.session = tf.Session(graph=prior_graph)
        with prior_graph.as_default():
            tf.initialize_all_variables().run(session=self.session)
        print("Initialized prior net")

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
        with prior_graph.as_default():
            tf.train.Saver().save(self.session, save_path)

    def validate_observations(self, dataset, labels):
        p, l = self.session.run([probs, loss], {X: dataset, Y: labels, keep_param: 1})
        a = accuracy(p, labels)
        print("Validation accuracy: %.1f%%" % a)
        print("Validation loss: %.4f" % l)
        accuracy_matrix(p, labels)

    def run_forward(self, data_point):
        dataset = [data_point]
        p = self.session.run([probs], {X: dataset, keep_param: 1})
        return p[0][0]

    def load(self, save_path):
        with prior_graph.as_default():
            tf.train.Saver().restore(self.session, save_path)

    def feed_reinforcement(self, board_arrays, moves, values, passes):
        for p in range(passes):
            p_boards, p_moves, p_values = permute(board_arrays, moves, values)
            for i in range(int(len(p_boards) / batch_size)):
                batch_data = p_boards[i * batch_size:(i + 1) * batch_size]
                batch_labels = moves[i * batch_size:(i + 1) * batch_size]
                batch_values = values[i * batch_size:(i + 1) * batch_size]
                feed_dict = {X: batch_data, Y: batch_labels, V: batch_values, keep_param: 1 - dropout}
                _, l, p, mid, pr, lo = self.session.run([reinforcement_optimizer, value, probs_for_actions, a, probs, logits], feed_dict=feed_dict)
        debug = 1