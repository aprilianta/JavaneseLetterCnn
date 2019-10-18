#!/usr/bin/env python

from __future__ import division

import argparse
import io
import math
import os
import random

import numpy as np
import tensorflow as tf

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

# Default path
DEFAULT_LABEL_CSV = os.path.join(SCRIPT_PATH, 'mycroft/labels-file.csv')
DEFAULT_LABEL_FILE = os.path.join(SCRIPT_PATH,
                                  'labels/label_aksara.txt')
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_PATH, 'tfrecords-dir-output')
DEFAULT_NUM_SHARDS_TRAIN = 4     #jumlah tf_record training yg akan dibuat
DEFAULT_NUM_SHARDS_TEST = 1      #jumlah tf_record test yg akan dibuat


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class TFRecordsConverter(object):
    """Class that handles converting images to TFRecords."""

    def __init__(self, labels_csv, label_file, output_dir,
                 num_shards_train, num_shards_test):

        self.output_dir = output_dir  #direktori output
        self.num_shards_train = num_shards_train #jumlah tf record training yg akan dibuat
        self.num_shards_test = num_shards_test #jumlah tf record test yg akan dibuat

        if not os.path.exists(self.output_dir): #buat direktori jika default path belum ada
            os.makedirs(self.output_dir)

        # Dapatkan list dari gambar dan label
        self.filenames, self.labels = \
            self.process_image_labels(labels_csv, label_file)   #labels_csv tembak csv default, label_file tembak .txt

        # Counter untuk total gambar yg diproses
        self.counter = 0

    def process_image_labels(self, labels_csv, label_file):
        """This will constuct two shuffled lists for images and labels.

        The index of each image in the images list will have the corresponding
        label at the same index in the labels list.
        """
        labels_csv = io.open(labels_csv, 'r', encoding='utf-8')   #read csv yg udh ditentukan lewat default path
        labels_file = io.open(label_file, 'r',
                              encoding='utf-8').read().splitlines() #read .txt (di .txt, pelabelan dipisahkan dengan \n)

        # Petakan karakter ke indeks
        label_dict = {}
        count = 0
        for label in labels_file:   #untuk setiap label dalam .txt ....
            label_dict[label] = count
            count += 1

        # Buat list untuk images dan label
        images = []
        labels = []
        for row in labels_csv:     #untuk tiap baris data dalam .csv .....
            file, label = row.strip().split(',')
            images.append(file)    #masukkan path berkas gambar ke list images
            labels.append(label_dict[label])   #masukkan label ke list labels

        # Acak urutan gambar dan label
        shuffled_indices = list(range(len(images)))
        random.seed(12121)      #kasih nilai awal random
        random.shuffle(shuffled_indices)
        filenames = [images[i] for i in shuffled_indices]
        labels = [labels[i] for i in shuffled_indices]

        return filenames, labels

    def write_tfrecords_file(self, output_path, indices):  #fungsi tulis tf record
        """Writes out TFRecords file."""
        writer = tf.python_io.TFRecordWriter(output_path) #buat tf record
        for i in indices:      #untuk indeks[i]
            filename = self.filenames[i]   #ambil filename
            label = self.labels[i]         #ambil label
            with tf.gfile.FastGFile(filename, 'rb') as f:    #return image bytes sebagai string
                im_data = f.read()

            # example adalah format data yang berisi penyimpanan nilai kunci, yang memetakan ke feature message.
            # setiap example berisi dua fitur.
            # Satu akan menjadi ByteList untuk data gambar mentah
            # dan yang lainnya akan menjadi Int64List yang berisi indeks label yang sesuai dalam daftar label dari file.

            example = tf.train.Example(features=tf.train.Features(feature={
                'image/class/label': _int64_feature(label),
                'image/encoded': _bytes_feature(tf.compat.as_bytes(im_data))}))
            writer.write(example.SerializeToString())
            self.counter += 1
            if not self.counter % 1000:
                print('Processed {} images...'.format(self.counter))
        writer.close()

    def convert(self):
        """This function will drive the conversion to TFRecords.

        Here, we partition the data into a training and testing set, then
        divide each data set into the specified number of TFRecords shards.
        """

        num_files_total = len(self.filenames) # total berkas gambar

        #alokasi 20% dr gambar untuk testing
        num_files_test = int(num_files_total * .20)

        #alokasi 80% dr gambar untuk training
        num_files_train = num_files_total - num_files_test

        print('Processing training set TFRecords...')

        files_per_shard = int(math.ceil(num_files_train /
                                        self.num_shards_train))    #dari alokasi train, bagi ke dalam bagian yg lebih kecil
        start = 0
        for i in range(1, self.num_shards_train):
            shard_path = os.path.join(self.output_dir,
                                      'train-{}.tfrecords'.format(str(i)))

            # Dapatkan subset indeks untuk mendapatkan hanya subset gambar / label untuk file shard saat ini.
            file_indices = np.arange(start, start+files_per_shard, dtype=int)     #indices = indeks
            start = start + files_per_shard
            self.write_tfrecords_file(shard_path, file_indices)

        # sisa gambar masuk ke final shard.
        file_indices = np.arange(start, num_files_train, dtype=int)
        final_shard_path = os.path.join(self.output_dir,
                                        'train-{}.tfrecords'.format(
                                            str(self.num_shards_train)))
        self.write_tfrecords_file(final_shard_path, file_indices)

        print('Processing testing set TFRecords...')

        files_per_shard = math.ceil(num_files_test / self.num_shards_test)
        start = num_files_train
        for i in range(1, self.num_shards_test):
            shard_path = os.path.join(self.output_dir,
                                      'test-{}.tfrecords'.format(str(i)))
            file_indices = np.arange(start, start+files_per_shard, dtype=int)
            start = start + files_per_shard
            self.write_tfrecords_file(shard_path, file_indices)

        # The remaining images will go in the final shard.
        file_indices = np.arange(start, num_files_total, dtype=int)
        final_shard_path = os.path.join(self.output_dir,
                                        'test-{}.tfrecords'.format(
                                            str(self.num_shards_test)))
        self.write_tfrecords_file(final_shard_path, file_indices)

        print('\nProcessed {} total images...'.format(self.counter))
        print('Number of training examples: {}'.format(num_files_train))
        print('Number of testing examples: {}'.format(num_files_test))
        print('TFRecords files saved to {}'.format(self.output_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-label-csv', type=str, dest='labels_csv',
                        default=DEFAULT_LABEL_CSV,
                        help='File containing image paths and corresponding '
                             'labels.')
    parser.add_argument('--label-file', type=str, dest='label_file',
                        default=DEFAULT_LABEL_FILE,
                        help='File containing newline delimited labels.')
    parser.add_argument('--output-dir', type=str, dest='output_dir',
                        default=DEFAULT_OUTPUT_DIR,
                        help='Output directory to store TFRecords files.')
    parser.add_argument('--num-shards-train', type=int,
                        dest='num_shards_train',
                        default=DEFAULT_NUM_SHARDS_TRAIN,
                        help='Number of shards to divide training set '
                             'TFRecords into.')
    parser.add_argument('--num-shards-test', type=int,
                        dest='num_shards_test',
                        default=DEFAULT_NUM_SHARDS_TEST,
                        help='Number of shards to divide testing set '
                             'TFRecords into.')
    args = parser.parse_args()
    converter = TFRecordsConverter(args.labels_csv,
                                   args.label_file,
                                   args.output_dir,
                                   args.num_shards_train,
                                   args.num_shards_test)
    converter.convert()
