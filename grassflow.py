import argparse
import os
import sys
import re
import tensorflow as tf
import numpy as np


tf.enable_eager_execution()
tf.logging.set_verbosity(tf.logging.ERROR)


def parse_flatten(args):
    return tf.keras.layers.Flatten(**args)


def parse_dropout(args):
    args['rate'] = float(args['rate'])
    return tf.keras.layers.Dropout(**args)


def parse_dense(args):
    units = int(args['units'])
    del args['units']
    return tf.keras.layers.Dense(units, **args)


PARSER = {
    "Flatten": parse_flatten,
    "Dropout": parse_dropout,
    "Dense": parse_dense,
}


def parse_optimizer(optimizer_line):
    parts = optimizer_line.split(':')
    name, parts = parts[0], parts[1:]
    opts = dict()
    for part in parts:
        k, v = part.split('=')
        if v.isnumeric():
            v = int(v)
        elif re.match('\\d+\\.\\d+', v):
            v = float(v)
        opts[k] = v
    clazz = tf.train.AdamOptimizer
    if name == 'adam':
        clazz = tf.train.AdamOptimizer
    elif name == 'rmsprop':
        clazz = tf.train.RMSPropOptimizer
    return clazz(**opts)


def process_model(epochs=5, train=True):
    layer_lines = sys.stdin.readline().strip()
    optimizer_line = sys.stdin.readline().strip()
    loss_line = sys.stdin.readline().strip()
    input_path = sys.stdin.readline().strip()

    layers = parse_layers(layer_lines)
    model = tf.keras.models.Sequential(layers)
    optimizer = parse_optimizer(optimizer_line)
    loss = loss_line.split(':')[1]
    model.compile(optimizer=optimizer, loss=loss, metrics=['mae'])
    if not os.path.isfile(os.path.join(input_path, "minmax.csv")):
        train = True

    if train:
        training_in, training_out, min_in, minmax_in, min_out, minmax_out = parse_input(input_path)
        model.fit(training_in, training_out, epochs=epochs, verbose=0)
        with open(os.path.join(input_path, "minmax.csv"), "w+") as stream:
            stream.write(",".join(str(i) for i in min_in) + "\n")
            stream.write(",".join(str(i) for i in minmax_in) + "\n")
            stream.write(",".join(str(i) for i in min_out) + "\n")
            stream.write(",".join(str(i) for i in minmax_out))
        model.save_weights(os.path.join(input_path, "weights"), overwrite=True)
    else:
        model.load_weights(os.path.join(input_path, "weights"))
        with open(os.path.join(input_path, "minmax.csv"), "r") as stream:
            rows = stream.read().split("\n")
            rows = [np.array([float(item) for item in row.split(',')]) for row in rows]
            min_in = rows[0]
            minmax_in = rows[1]
            min_out = rows[2]
            minmax_out = rows[3]

    while True:
        line = sys.stdin.readline()
        if not line:
            break
        input_data = line.strip()
        input_data = np.array([[float(item) for item in input_data.split(',')]])
        feed = (input_data - min_in) / minmax_in
        results = model.predict(feed) * minmax_out + min_out
        result = ""
        for row in results:
            result = ",".join(str(item) for item in row)
        print(result)
        sys.stdout.flush()


def parse_layers(layer_lines):
    layer_lines = layer_lines.split(',')
    layers = []
    for line in layer_lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split(":")
        if not parts:
            continue
        parser_name = parts[0]
        if parser_name not in PARSER:
            continue
        parser = PARSER[parser_name]
        opts = dict()
        for pair in parts[1:]:
            opts[pair.split('=')[0]] = pair.split('=')[1]
        layer = parser(opts)
        layers.append(layer)
    return layers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--epochs', default=5, type=int)
    opts = parser.parse_args()
    process_model(train=opts.train, epochs=opts.epochs)


def parse_input(path):
    path = os.path.join(path, 'data.csv')
    data = open(path, "r").read().split("\n")
    num_in = int(data[1].split('=')[1].strip())
    data = [[float(val) for val in line.split(',')] for line in data[3:] if line]
    training_data = np.array(data).reshape((len(data), len(data[0])))
    training_in = training_data[:, 1:num_in+1]
    training_out = training_data[:, num_in+1:]

    min_in, minmax_in = np.min(training_in, axis=0), np.max(training_in, axis=0) - np.min(training_in, axis=0)
    training_in = (training_in - min_in) / minmax_in
    min_out, minmax_out = np.min(training_out, axis=0), np.max(training_out, axis=0) - np.min(training_out, axis=0)
    minmax_out[minmax_out == 0] = 1.0
    training_out = (training_out - min_out) / minmax_out
    return training_in, training_out, min_in, minmax_in, min_out, minmax_out


if __name__ == '__main__':
    main()
