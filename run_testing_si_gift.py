# coding=utf-8
"""MPC-BERT testing runner on the downstream task of speaker identification."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import operator
from time import time
from collections import defaultdict
import tensorflow as tf
import optimization
import tokenization
import modeling_speaker_gift as modeling

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("task_name", 'Testing', 
                    "The name of the task.")

flags.DEFINE_string("test_dir", 'test.tfrecord',
                    "The input test data dir. Should contain the .tsv files (or other data files) for the task.")

flags.DEFINE_string("restore_model_dir", 'output/',
                    "The output directory where the model checkpoints have been written.")

flags.DEFINE_string("bert_config_file", 'uncased_L-12_H-768_A-12/bert_config.json',
                    "The config json file corresponding to the pre-trained BERT model. "
                    "This specifies the model architecture.")

flags.DEFINE_bool("do_eval", True, 
                  "Whether to run eval on the dev set.")

flags.DEFINE_integer("eval_batch_size", 32, 
                     "Total batch size for predict.")

flags.DEFINE_integer("max_seq_length", 320,
                     "The maximum total input sequence length after WordPiece tokenization. "
                     "Sequences longer than this will be truncated, and sequences shorter "
                     "than this will be padded.")

flags.DEFINE_integer("max_utr_num", 7, 
                     "Maximum utterance number.")


def print_configuration_op(FLAGS):
    print('My Configurations:')
    for name, value in FLAGS.__flags.items():
        value=value.value
        if type(value) == float:
            print(' %s:\t %f'%(name, value))
        elif type(value) == int:
            print(' %s:\t %d'%(name, value))
        elif type(value) == str:
            print(' %s:\t %s'%(name, value))
        elif type(value) == bool:
            print(' %s:\t %s'%(name, value))
        else:
            print('%s:\t %s' % (name, value))
    print('End of configuration')


def count_data_size(file_name):
    sample_nums = 0
    for record in tf.python_io.tf_record_iterator(file_name):
        sample_nums += 1
    return  sample_nums


def parse_exmp(serial_exmp):
    input_data = tf.parse_single_example(serial_exmp,
                                       features={
                                           "input_sents":
                                               tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
                                           "input_mask":
                                               tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
                                           "segment_ids":
                                               tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
                                           "speaker_ids":
                                               tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
                                           "cls_positions":
                                               tf.FixedLenFeature([FLAGS.max_utr_num], tf.int64),
                                           "rsp_position":
                                               tf.FixedLenFeature([1], tf.int64),
                                           "label_ids":
                                               tf.FixedLenFeature([FLAGS.max_utr_num], tf.int64),
                                           "reply_mask_utr2word_flatten":
                                               tf.FixedLenFeature([FLAGS.max_utr_num * FLAGS.max_seq_length], tf.int64),
                                           "utr_lens":
                                               tf.FixedLenFeature([FLAGS.max_utr_num], tf.int64),
                                       }
                                       )
    # So cast all int64 to int32.
    for name in list(input_data.keys()):
        t = input_data[name]
        if t.dtype == tf.int64:
            t = tf.to_int32(t)
        input_data[name] = t

    input_sents = input_data["input_sents"]
    input_mask = input_data["input_mask"]
    segment_ids= input_data["segment_ids"]
    speaker_ids= input_data["speaker_ids"]
    cls_positions= input_data["cls_positions"]
    rsp_position= input_data["rsp_position"]
    reply_mask_utr2word = tf.reshape(input_data['reply_mask_utr2word_flatten'], [FLAGS.max_utr_num, FLAGS.max_seq_length])
    utr_lens = input_data['utr_lens']
    labels = input_data['label_ids']

    reply_mask_word2word = []
    for i in range(FLAGS.max_utr_num):
        reply_mask_utr2word_i = reply_mask_utr2word[i]  # [max_seq_length, ]
        utr_len_i = utr_lens[i]                         # [1, ]
        reply_mask_utr2word_i_tiled = tf.tile(tf.expand_dims(reply_mask_utr2word_i, 0), [utr_len_i, 1])  # [utr_len, max_seq_length]
        reply_mask_word2word.append(reply_mask_utr2word_i_tiled)
    
    reply_mask_pad = tf.zeros(shape=[FLAGS.max_seq_length - tf.reduce_sum(utr_lens), FLAGS.max_seq_length], dtype=tf.int32)
    reply_mask_word2word.append(reply_mask_pad)
    reply_mask = tf.concat(reply_mask_word2word, 0)     # [max_seq_length, max_seq_length]
    print("* Loading reply mask ...")
    
    return input_sents, input_mask, segment_ids, speaker_ids, cls_positions, rsp_position, labels, reply_mask


def gather_indexes(sequence_tensor, positions):
    """Gathers the vectors at the specific positions over a minibatch."""
    # sequence_tensor = [batch_size, seq_length, width]
    # positions = [batch_size, max_utr_num]
    sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    flat_offsets = tf.reshape(
        tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])  # [batch_size, 1]
    flat_positions = tf.reshape(positions + flat_offsets, [-1])         # [batch_size*max_utr_num, ]
    flat_sequence_tensor = tf.reshape(sequence_tensor,
                                    [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)     # [batch_size*max_utr_num, width]
    return output_tensor


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids, speaker_ids, cls_positions, rsp_position, labels, reply_mask, 
                 num_labels, use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        speaker_ids=speaker_ids,
        reply_mask=reply_mask,
        use_one_hot_embeddings=use_one_hot_embeddings)

    input_tensor = gather_indexes(model.get_sequence_output(), cls_positions)  # [batch_size*max_utr_num, dim]

    input_shape = modeling.get_shape_list(input_tensor, expected_rank=2)
    width = input_shape[-1]
    positions_shape = modeling.get_shape_list(cls_positions, expected_rank=2)
    max_utr_num = positions_shape[-1]

    with tf.variable_scope("cls/speaker_restore"):
        # We apply one more non-linear transformation before the output layer.
        with tf.variable_scope("transform"):
            input_tensor = tf.layers.dense(
                input_tensor,
                units=bert_config.hidden_size,
                activation=modeling.get_activation(bert_config.hidden_act),
                kernel_initializer=modeling.create_initializer(bert_config.initializer_range))
            input_tensor = modeling.layer_norm(input_tensor)  # [batch_size*max_utr_num, dim]

        input_tensor = tf.reshape(input_tensor, [-1, max_utr_num, width])  # [batch_size, max_utr_num, dim]

        rsp_tensor = gather_indexes(input_tensor, rsp_position)  # [batch_size*1, dim]
        rsp_tensor = tf.reshape(rsp_tensor, [-1, 1, width])      # [batch_size, 1, dim]

        output_weights = tf.get_variable(
            "output_weights",
            shape=[width, width],
            initializer=modeling.create_initializer(bert_config.initializer_range))
        logits = tf.matmul(tf.einsum('aij,jk->aik', rsp_tensor, output_weights),
                            input_tensor, transpose_b=True)                # [batch_size, 1, max_utr_num]
        logits = tf.squeeze(logits, [1])                                   # [batch_size, max_utr_num]

        mask = tf.sequence_mask(tf.reshape(rsp_position, [-1, ]), max_utr_num, dtype=tf.float32)  # [batch_size, max_utr_num]
        logits = logits * mask + -1e9 * (1-mask)                   
        log_probs = tf.nn.log_softmax(logits, axis=-1)                     # [batch_size, max_utr_num]

        # loss
        one_hot_labels = tf.cast(labels, "float")                                    # [batch_size, max_utr_num]
        per_example_loss = - tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])  # [batch_size, ]
        mean_loss = tf.reduce_mean(per_example_loss, name="mean_loss")

        # accuracy
        predictions = tf.argmax(log_probs, axis=-1, output_type=tf.int32)                    # [batch_size, ]
        predictions_one_hot = tf.one_hot(predictions, depth=max_utr_num, dtype=tf.float32)   # [batch_size, max_utr_num]
        correct_prediction = tf.reduce_sum(predictions_one_hot * one_hot_labels, -1)         # [batch_size, ]
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name="accuracy")

    return mean_loss, logits, log_probs, accuracy


def run_test(sess, training, prob, accuracy):

    step = 0
    t0 = time()
    num_test = 0
    num_correct = 0.0
    test_accuracy = 0

    try:
        while True:
            step += 1
            batch_accuracy, predicted_prob = sess.run([accuracy, prob], feed_dict={training: False})
            
            num_test += len(predicted_prob)
            num_correct += len(predicted_prob) * batch_accuracy

            if step % 100 == 0:
                tf.logging.info("Step %d, Time (min): %.2f" % (step, (time() - t0) / 60.0))

    except tf.errors.OutOfRangeError:
        test_accuracy = num_correct / num_test
        print('num_test_samples: {}, test_accuracy: {}'.format(num_test, test_accuracy))
        
    return test_accuracy


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    print_configuration_op(FLAGS)

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    test_data_size = count_data_size(FLAGS.test_dir)
    tf.logging.info('test data size: {}'.format(test_data_size))

    filenames = tf.placeholder(tf.string, shape=[None])
    shuffle_size = tf.placeholder(tf.int64)
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse_exmp)  # Parse the record into tensors.
    dataset = dataset.repeat(1)
    # dataset = dataset.shuffle(shuffle_size)
    dataset = dataset.batch(FLAGS.eval_batch_size)
    iterator = dataset.make_initializable_iterator()
    input_sents, input_mask, segment_ids, speaker_ids, cls_positions, rsp_position, labels, reply_mask = iterator.get_next()

    training = tf.placeholder(tf.bool)
    mean_loss, logits, log_probs, accuracy = create_model(bert_config = bert_config,
                                                             is_training = training,
                                                             input_ids = input_sents,
                                                             input_mask = input_mask,
                                                             segment_ids = segment_ids,
                                                             speaker_ids = speaker_ids,
                                                             cls_positions = cls_positions,
                                                             rsp_position = rsp_position,
                                                             reply_mask = reply_mask,
                                                             labels = labels,
                                                             num_labels = 1,
                                                             use_one_hot_embeddings = False)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    if FLAGS.do_eval:
        with tf.Session(config=config) as sess:
            tf.logging.info("*** Restore model ***")

            ckpt = tf.train.get_checkpoint_state(FLAGS.restore_model_dir)
            variables = tf.trainable_variables()
            saver = tf.train.Saver(variables)
            saver.restore(sess, ckpt.model_checkpoint_path)

            tf.logging.info('Test begin')
            sess.run(iterator.initializer,
                     feed_dict={filenames: [FLAGS.test_dir], shuffle_size: 1})
            run_test(sess, training, log_probs, accuracy)


if __name__ == "__main__":
    tf.app.run()
