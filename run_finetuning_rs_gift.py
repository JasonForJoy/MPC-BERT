# coding=utf-8
"""MPC-BERT finetuning runner on the downstream task of response selection."""

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
import metrics

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("train_dir", 'train.tfrecord',
                    "The input train data dir. Should contain the .tsv files (or other data files) for the task.")

flags.DEFINE_string("valid_dir", 'valid.tfrecord',
                    "The input valid data dir. Should contain the .tsv files (or other data files) for the task.")

flags.DEFINE_string("output_dir", 'output',
                    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string("task_name", 'ResponseSelection', 
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
                                           "ctx_id":
                                               tf.FixedLenFeature([], tf.int64),
                                           "rsp_id":
                                               tf.FixedLenFeature([], tf.int64),
                                           "input_sents":
                                               tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
                                           "input_mask":
                                               tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
                                           "segment_ids":
                                               tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
                                           "speaker_ids":
                                               tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
                                           "reply_mask_utr2word_flatten":
                                               tf.FixedLenFeature([FLAGS.max_utr_num * FLAGS.max_seq_length], tf.int64),
                                           "utr_lens":
                                               tf.FixedLenFeature([FLAGS.max_utr_num], tf.int64),
                                           "label_ids":
                                               tf.FixedLenFeature([], tf.float32),
                                       }
                                       )
    # So cast all int64 to int32.
    for name in list(input_data.keys()):
        t = input_data[name]
        if t.dtype == tf.int64:
            t = tf.to_int32(t)
        input_data[name] = t

    ctx_id = input_data["ctx_id"]
    rsp_id = input_data['rsp_id']
    input_sents = input_data["input_sents"]
    input_mask = input_data["input_mask"]
    segment_ids= input_data["segment_ids"]
    speaker_ids= input_data["speaker_ids"]
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
    
    return ctx_id, rsp_id, input_sents, input_mask, segment_ids, speaker_ids, reply_mask, labels


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids, speaker_ids, reply_mask, labels, ctx_id, rsp_id,
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

    target_loss_weight = [1.0, 1.0]
    target_loss_weight = tf.convert_to_tensor(target_loss_weight)

    flagx = tf.cast(tf.greater(labels, 0), dtype=tf.float32)
    flagy = tf.cast(tf.equal(labels, 0), dtype=tf.float32)

    all_target_loss = target_loss_weight[1] * flagx + target_loss_weight[0] * flagy

    output_layer = model.get_pooled_output()
    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))
    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        output_layer = tf.layers.dropout(output_layer, rate=0.1, training=is_training)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)

        probabilities = tf.sigmoid(logits, name="prob")
        logits = tf.squeeze(logits,[1])
        losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
        losses = tf.multiply(losses, all_target_loss)

        mean_loss = tf.reduce_mean(losses, name="mean_loss") + sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(tf.sign(probabilities - 0.5), tf.sign(labels - 0.5))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name="accuracy")

    return mean_loss, logits, probabilities, accuracy


def run_epoch(epoch, op_name, sess, training, logits, accuracy, mean_loss, train_opt):

    step = 0
    t0 = time()

    try:
        while True:
            step += 1
            batch_logits, batch_loss, _, batch_accuracy = sess.run([logits, mean_loss, train_opt, accuracy], feed_dict={training:True})

            if step % 1000 == 0:
                tf.logging.info("Epoch: %i, Step: %d, Time (min): %.2f, Loss: %.4f, Accuracy: %.2f" %
                                  (epoch, step, (time() - t0) / 60.0, batch_loss, 100 * batch_accuracy))

    except tf.errors.OutOfRangeError:
        tf.logging.info("Epoch: %i, Step: %d, Time (min): %.2f, Loss: %.4f, Accuracy: %.2f" %
                          (epoch, step, (time() - t0) / 60.0, batch_loss, 100 * batch_accuracy))
        pass


best_score = 0.0
def run_test(epoch, op_name, sess, training, accuracy, prob, pair_ids, saver, dir_path):

    step = 0
    t0 = time()
    num_test = 0
    num_correct = 0.0
    mrr = 0
    results = defaultdict(list)
    
    try:
        while True:
            step += 1
            batch_accuracy, predicted_prob, pair_ = sess.run([accuracy, prob, pair_ids], feed_dict={training:False})
            question_id, answer_id, label = pair_

            num_test += len(predicted_prob)
            num_correct += len(predicted_prob) * batch_accuracy
            for i, prob_score in enumerate(predicted_prob):
                results[question_id[i]].append((answer_id[i], label[i], prob_score[0]))

            if step % 1000 == 0:
                tf.logging.info("Epoch: %i, Step: %d, Time (min): %.2f" % (epoch, step, (time() - t0)/60.0 ))

    except tf.errors.OutOfRangeError:
        print('num_test_samples: {}, test_accuracy: {}'.format(num_test, num_correct / num_test))
        accu, precision, recall, f1, loss = metrics.classification_metrics(results)
        print('Accuracy: {}, Precision: {}, Recall: {}, F1: {}, Loss: {}'.format(accu, precision, recall, f1, loss))

        mvp = metrics.mean_average_precision(results)
        mrr = metrics.mean_reciprocal_rank(results)
        top_1_precision = metrics.top_1_precision(results)
        total_valid_query = metrics.get_num_valid_query(results)
        print('MAP (mean average precision: {}, MRR (mean reciprocal rank): {}, Top-1 precision: {}, Num_query: {}'.format(
            mvp, mrr, top_1_precision, total_valid_query))

        out_path = os.path.join(dir_path, "output_epoch_{}.txt".format(epoch))
        print("Saving evaluation to {}".format(out_path))
        with open(out_path, 'w') as f:
            f.write("query_id\tdocument_id\tscore\trank\trelevance\n")
            for us_id, v in results.items():
                v.sort(key=operator.itemgetter(2), reverse=True)
                for i, rec in enumerate(v):
                    r_id, label, prob_score = rec
                    rank = i + 1
                    f.write('{}\t{}\t{}\t{}\t{}\n'.format(us_id, r_id, prob_score, rank, label))       

        global best_score
        if op_name == 'valid' and mrr > best_score:
            best_score = mrr
            dir_path = os.path.join(dir_path, "epoch_{}".format(epoch))
            saver.save(sess, dir_path)
            tf.logging.info(">> Save model!")

    return mrr



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
    ctx_id, rsp_id, input_sents, input_mask, segment_ids, speaker_ids, reply_mask, labels = iterator.get_next()
    pair_ids = [ctx_id, rsp_id, labels]


    training = tf.placeholder(tf.bool)
    mean_loss, logits, probabilities, accuracy = create_model(bert_config = bert_config,
                                                               is_training = training,
                                                               input_ids = input_sents,
                                                               input_mask = input_mask,
                                                               segment_ids = segment_ids,
                                                               speaker_ids = speaker_ids,
                                                               reply_mask = reply_mask,
                                                               labels = labels,
                                                               ctx_id = ctx_id,
                                                               rsp_id = rsp_id,
                                                               num_labels = 1,
                                                               use_one_hot_embeddings = False)

    # init model with pre-training
    tvars = tf.trainable_variables()
    if FLAGS.init_checkpoint:
        (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,FLAGS.init_checkpoint)
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
                run_test(epoch, "valid", sess, training, accuracy, probabilities, pair_ids, saver, root_path)


if __name__ == "__main__":
    tf.app.run()
