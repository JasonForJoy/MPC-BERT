# coding=utf-8
"""MPC-BERT finetuning runner on the downstream task of speaker identification."""

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

flags.DEFINE_string("train_dir", 'train.tfrecord',
                    "The input train data dir. Should contain the .tsv files (or other data files) for the task.")

flags.DEFINE_string("valid_dir", 'valid.tfrecord',
                    "The input valid data dir. Should contain the .tsv files (or other data files) for the task.")

flags.DEFINE_string("output_dir", 'output',
                    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string("task_name", 'SpeakerIdentification', 
                    "The name of the task to train.")

flags.DEFINE_string("bert_config_file", 'uncased_L-12_H-768_A-12/bert_config.json',
                    "The config json file corresponding to the pre-trained BERT model. "
                    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", 'uncased_L-12_H-768_A-12/vocab.txt',
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string("init_checkpoint", 'uncased_L-12_H-768_A-12/bert_model.ckpt',
                    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool("do_lower_case", True,
                  "Whether to lower case the input text. Should be True for uncased "
                  "models and False for cased models.")

flags.DEFINE_integer("max_seq_length", 320,
                     "The maximum total input sequence length after WordPiece tokenization. "
                     "Sequences longer than this will be truncated, and sequences shorter "
                     "than this will be padded.")

flags.DEFINE_integer("max_utr_num", 7, 
                     "Maximum utterance number.")

flags.DEFINE_bool("do_train", True, 
                  "Whether to run training.")

flags.DEFINE_float("warmup_proportion", 0.1,
                   "Proportion of training to perform linear learning rate warmup for. "
                   "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("train_batch_size", 12, 
                     "Total batch size for training.")

flags.DEFINE_float("learning_rate", 2e-5, 
                   "The initial learning rate for Adam.")

flags.DEFINE_integer("num_train_epochs", 5, 
                     "Total number of training epochs to perform.")



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
    return sample_nums


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
                                           "reply_mask_utr2word_flatten":
                                               tf.FixedLenFeature([FLAGS.max_utr_num * FLAGS.max_seq_length], tf.int64),
                                           "utr_lens":
                                               tf.FixedLenFeature([FLAGS.max_utr_num], tf.int64),
                                           "label_ids":
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

    return input_sents, input_mask, segment_ids, speaker_ids, cls_positions, rsp_position, reply_mask, labels


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


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids, speaker_ids, cls_positions, rsp_position, reply_mask, labels, use_one_hot_embeddings):
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


def run_epoch(epoch, op_name, sess, training, logits, accuracy, mean_loss, train_opt):

    step = 0
    t0 = time()

    try:
        while True:
            step += 1
            batch_logits, batch_loss, _, batch_accuracy = sess.run([logits, mean_loss, train_opt, accuracy], feed_dict={training: True})

            if step % 1000 == 0:
                tf.logging.info("Epoch: %i, Step: %d, Time (min): %.2f, Loss: %.4f, Accuracy: %.2f" %
                                  (epoch, step, (time() - t0) / 60.0, batch_loss, 100 * batch_accuracy))

    except tf.errors.OutOfRangeError:
        tf.logging.info("Epoch: %i, Step: %d, Time (min): %.2f, Loss: %.4f, Accuracy: %.2f" %
                          (epoch, step, (time() - t0) / 60.0, batch_loss, 100 * batch_accuracy))
        pass


best_score = 0.0
def run_test(epoch, op_name, sess, training, prob, accuracy, saver, dir_path):

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
                tf.logging.info("Epoch: %i, Step: %d, Time (min): %.2f" % (epoch, step, (time() - t0)/60.0 ))

    except tf.errors.OutOfRangeError:
        test_accuracy = num_correct / num_test
        print('num_test_samples: {}, test_accuracy: {}'.format(num_test, test_accuracy))     

        global best_score
        if op_name == 'valid' and test_accuracy > best_score:
            best_score = test_accuracy
            dir_path = os.path.join(dir_path, "epoch_{}".format(epoch))
            saver.save(sess, dir_path)
            tf.logging.info(">> Save model!")

    return test_accuracy



def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    print_configuration_op(FLAGS)

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    root_path = FLAGS.output_dir
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    timestamp = str(int(time()))
    root_path = os.path.join(root_path, timestamp)
    tf.logging.info('root_path: {}'.format(root_path))
    if not os.path.exists(root_path):
        os.makedirs(root_path)

    train_data_size = count_data_size(FLAGS.train_dir)
    tf.logging.info('train data size: {}'.format(train_data_size))
    valid_data_size = count_data_size(FLAGS.valid_dir)
    tf.logging.info('valid data size: {}'.format(valid_data_size))

    num_train_steps = train_data_size // FLAGS.train_batch_size * FLAGS.num_train_epochs
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    filenames = tf.placeholder(tf.string, shape=[None])
    shuffle_size = tf.placeholder(tf.int64)
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse_exmp)  # Parse the record into tensors.
    dataset = dataset.repeat(1)
    # buffer_size 100
    dataset = dataset.shuffle(shuffle_size)
    dataset = dataset.batch(FLAGS.train_batch_size)
    iterator = dataset.make_initializable_iterator()
    input_sents, input_mask, segment_ids, speaker_ids, cls_positions, rsp_position, reply_mask, labels = iterator.get_next()


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
                                                             use_one_hot_embeddings = False)

    # init model with pre-training
    tvars = tf.trainable_variables()
    if FLAGS.init_checkpoint:
        (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, FLAGS.init_checkpoint)
        tf.train.init_from_checkpoint(FLAGS.init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
        init_string = ""
        if var.name in initialized_variable_names:
            init_string = ", *INIT_FROM_CKPT*"
        tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

    train_opt = optimization.create_optimizer(mean_loss, FLAGS.learning_rate, num_train_steps, num_warmup_steps, False)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    saver = tf.train.Saver()

    if FLAGS.do_train:
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(FLAGS.num_train_epochs):
                tf.logging.info('Train begin epoch {}'.format(epoch))
                sess.run(iterator.initializer,
                         feed_dict={filenames: [FLAGS.train_dir], shuffle_size: 1024})
                run_epoch(epoch, "train", sess, training, logits, accuracy, mean_loss, train_opt)

                tf.logging.info('Valid begin')
                sess.run(iterator.initializer,
                         feed_dict={filenames: [FLAGS.valid_dir], shuffle_size: 1})
                run_test(epoch, "valid", sess, training, log_probs, accuracy, saver, root_path)


if __name__ == "__main__":
    tf.app.run()
