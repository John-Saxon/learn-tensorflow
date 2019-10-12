
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import logging
# from modeltrain.config.data_config import DataConfig

try:
    from tensorflow.contrib.data.python.ops.readers import ORCFileDataset, MagmaDataset
except ImportError as e:
    logging.warning("cannot import ORCFileDataset or MagmaDataset")


def url_parse(url):
    if not isinstance(url, str):
        ValueError("url must be str type")
    post = url.index("://")
    protocol = url[:post]
    last = url[post + 3:]
    post = last.index("/")
    addr = last[:post]
    path = last[post + 1:]
    return protocol, addr, path


def hdfs_connection(url):
    protocol, addr, path = url_parse(url)
    namenode = protocol + "://" + addr + "/"
    try:
        tf.gfile.Stat(namenode)
        return True
    except Exception as e:
        print(e)
        return False


def get_data_file(path):
    """
        Return paths of param:path or files under param:path
    """
    if path.startwith("hdfs://") and not hdfs_connection(path):
        raise ValueError("cannot connect to hdfs %s" % path)
    if path.startwith("magma://"):
        return path
    files = []
    if not tf.gfile.Exists(path):
        raise ValueError("file path %s not exists" % path)
    if path and tf.gfile.IsDirectory(path):
        for i in tf.gfile.ListDirectory(path):
            fileStr = os.path.join(path, i)
            if not tf.gfile.IsDirectory(fileStr):
                files.append(fileStr)
    elif path:
        files.append(path)
    return files


class DataConverter(object):
    def __init__(self, config):
        self.config = config
        self.data_config = DataConfig(self.config)

    def input_func(self, files,
                   batch_size,
                   data_format="orc",
                   num_epochs=None,
                   shuffle=True,
                   without_label=False,
                   kind="regression",
                   tf_config=None):
        def data_decode(record):
            if without_label:
                if data_format == "orc" or data_format.startswith("magma"):
                    features = self.data_config.get_features_from_tf_record(
                        record)
                elif data_format == "csv":
                    features = self.data_config.get_features_from_csv(record)
                else:
                    raise TypeError(
                        "data format %s not support yet." % data_format)
                return features
            else:
                if data_format == "orc" or data_format.startswith("magma"):
                    features, labels = self.data_config.get_features_labels_from_tf_record(
                        record)
                elif data_format == "csv":
                    features, labels = self.data_config.get_features_labels_from_csv(
                        record)
                else:
                    raise TypeError(
                        "data format %s not support yet." % data_format)
                dimension = self.data_config.get_label_dimension()
                if kind == "classify":
                    if dimension != 1:
                        raise ValueError(
                            "label dimension error, when kind is classify label dimension must be 1")
                return features, labels

        if data_format == "orc":
            dataset = ORCFileDataset(files)
        elif data_format == "csv":
            dataset = tf.data.TextLineDataset(files)
        elif data_format.startswith("magma"):
            dataset = MagmaDataset(files)
        else:
            raise TypeError("data format %s not support yet." % data_format)
        buff_size = 10000 + 3 * batch_size
        if tf_config:
            num_worker = len(tf_config["cluster"]["worker"])
            num_chief = len(tf_config["cluster"]["chief"])
            if tf_config["task"]["type"] == "chief":
                dataset.shard(num_worker + num_chief,
                              tf_config["task"]["index"])
            else:
                dataset.shard(num_worker + num_chief,
                              tf_config["task"]["index"] + num_chief)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=buff_size)
        dataset = dataset.batch(batch_size=batch_size)
        if num_epochs and shuffle:
            dataset = dataset.repeat(num_epochs)
        dataset = dataset.map(data_decode)
        return dataset

    def get_feature_columns(self):
        return self.data_config.get_one_hot_feature_columns()

    def bt_get_feature_columns(self):
        return self.data_config.bt_get_bucket_or_indicator_feature_columns()

    def get_linear_feature_columns(self):
        return self.data_config.get_linear_feature_columns()

    def get_label_dimension(self):
        return self.data_config.get_label_dimension()

    def get_predict_features(self):
        return self.data_config.get_predict_features()
