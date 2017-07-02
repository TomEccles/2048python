import numpy as np
import tensorflow as tf

from file_utils import create_dir_if_needed

num_inputs = 16 + 16 + 16 * 16 + 4
num_outputs = 4
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
    shuffled_labels = labels[permutation, :]
    return shuffled_dataset, shuffled_labels


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


entropy_graph = tf.Graph()


def calc(x, keep_param=1):
    mat_1 = tf.matmul(x, weights_1) + biases_1
    rel_1 = tf.nn.relu(mat_1)
    drop_1 = tf.nn.dropout(rel_1, keep_param)
    # mat_2 = tf.matmul(drop_1, weights_2) + biases_2
    # rel_2 = tf.nn.relu(mat_2)
    # drop_2 = tf.nn.dropout(rel_2, keep_param)
    return tf.matmul(drop_1, weights_3) + biases_3


with entropy_graph.as_default():
    X = tf.placeholder(tf.float32, shape=(batch_size, num_inputs))
    Y = tf.placeholder(tf.float32, shape=(batch_size, num_outputs))
    keep_param = tf.placeholder(tf.float32)
    learning_rate = tf.placeholder(tf.float32)

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
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))

    # Optimizer.
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

    # Run it forwards
    forwards_input = tf.placeholder(tf.float32, shape=(None, num_inputs))
    forwards_pred = tf.nn.softmax(calc(forwards_input))

    # When we want prediction error
    forwards_labels = tf.placeholder(tf.float32, shape=(None, num_outputs))
    pred_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=forwards_pred, labels=forwards_labels))

checkpoint = "predictor.ckpt"
session = tf.Session(graph=entropy_graph)
with entropy_graph.as_default():
    tf.initialize_all_variables().run(session=session)
print("Initialized")


def feed_observations(dataset, labels, passes):
    for p in range(passes):
        permuted_data, permuted_labels = permute(dataset, labels)
        for i in range(int(len(dataset) / batch_size)):
            batch_data = permuted_data[i * batch_size:(i + 1) * batch_size]
            batch_labels = permuted_labels[i * batch_size:(i + 1) * batch_size]
            feed_dict = {X: batch_data, Y: batch_labels, keep_param: 1 - dropout}
            _, l = session.run([optimizer, loss], feed_dict=feed_dict)


def save(save_path):
    create_dir_if_needed(save_path)
    with entropy_graph.as_default():
        tf.train.Saver().save(session, save_path)


def validate_observations(dataset, labels):
    p, l = session.run([forwards_pred, pred_loss], {forwards_input: dataset, forwards_labels: labels})
    a = accuracy(p, labels)
    print("Validation accuracy: %.1f%%" % a)
    print("Validation loss: %.4f" % l)
    accuracy_matrix(p, labels)


def run_forward(data_point):
    p = session.run([forwards_pred], {forwards_input: [data_point]})
    return p[0][0]


def load(save_path):
    with entropy_graph.as_default():
        tf.train.Saver().restore(session, save_path)
