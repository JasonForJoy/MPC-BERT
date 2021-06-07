# coding=utf-8
"""Run (MLM + NSP + RUR + ISS + PCD + MSUR + SND) pre-training for MPC-BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import datetime
from time import time
import tensorflow as tf
import optimization
import modeling_speaker as modeling

flags = tf.flags
FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string("task_name", 'MPC-BERT Pre-training', 
                    "The name of the task to train.")

flags.DEFINE_string("input_file", './data/ijcai2019/pretraining_data.tfrecord',
                    "The input data dir. Should contain the .tsv files (or other data files) for the task.")

flags.DEFINE_string("output_dir", './uncased_L-12_H-768_A-12_pretrained',
                    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string("bert_config_file", 'uncased_L-12_H-768_A-12/bert_config.json',
                    "The config json file corresponding to the pre-trained BERT model. "
                    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", 'uncased_L-12_H-768_A-12/vocab.txt',
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string("init_checkpoint", 'uncased_L-12_H-768_A-12/bert_model.ckpt',
                    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool("do_train", True, 
                  "Whether to run training.")

# flags.DEFINE_bool("do_eval", True, 
#                   "Whether to run eval on the dev set.")

flags.DEFINE_integer("mid_save_step", 20000,
                     "Epoch is so long, mid_save_step 15000 is roughly 3 hours")

flags.DEFINE_integer("max_seq_length", 230,
                     "The maximum total input sequence length after WordPiece tokenization. "
                     "Sequences longer than this will be truncated, and sequences shorter "
                     "than this will be padded. Must match data generation.")

flags.DEFINE_integer("max_utr_length", 30, 
                     "Maximum single utterance length.")

flags.DEFINE_integer("max_utr_num", 7, 
                     "Maximum utterance number.")

flags.DEFINE_integer("max_predictions_per_seq", 25,
                     "Maximum number of masked LM predictions per sequence. "
                     "Must match data generation.")

flags.DEFINE_integer("max_predictions_per_seq_ar", 4,
                     "Maximum number of Reply-to Utterance Recognition (RUR) predictions per sequence.")

flags.DEFINE_integer("max_predictions_per_seq_sr", 2,
                     "Maximum number of Identical Speaker Searching (ISS) predictions per sequence.")

flags.DEFINE_integer("max_predictions_per_seq_cd", 2,
                     "Maximum number of Pointer Consistency Distinction (PCD) predictions per sequence.")

flags.DEFINE_integer("train_batch_size", 16, 
                     "Total batch size for training.")

# flags.DEFINE_integer("eval_batch_size", 8, 
#                      "Total batch size for eval.")

flags.DEFINE_float("learning_rate", 5e-5, 
                   "The initial learning rate for Adam.")

flags.DEFINE_float("warmup_proportion", 0.1, 
                   "Number of warmup steps.")

flags.DEFINE_integer("num_train_epochs", 10, 
                     "num_train_epochs.")


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
                                           "input_ids_mlm_nsp":
                                               tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
                                           "input_mask_mlm_nsp":
                                               tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
                                           "segment_ids_mlm_nsp":
                                               tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
                                           "speaker_ids_mlm_nsp":
                                               tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
                                           "next_sentence_labels":
                                               tf.FixedLenFeature([1], tf.int64),
                                           "masked_lm_positions":
                                               tf.FixedLenFeature([FLAGS.max_predictions_per_seq], tf.int64),
                                           "masked_lm_ids":
                                               tf.FixedLenFeature([FLAGS.max_predictions_per_seq], tf.int64),
                                           "masked_lm_weights":
                                               tf.FixedLenFeature([FLAGS.max_predictions_per_seq], tf.float32),
                                           
                                           "cls_positions":
                                               tf.FixedLenFeature([FLAGS.max_utr_num], tf.int64),
                                           "input_ids_ar_msr_pcd":
                                               tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
                                           "input_mask_ar_msr_pcd":
                                               tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
                                           "segment_ids_ar_msr_pcd":
                                               tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
                                           "speaker_ids_ar_msr_pcd":
                                               tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),

                                           "adr_recog_positions":
                                               tf.FixedLenFeature([FLAGS.max_predictions_per_seq_ar], tf.int64),
                                           "adr_recog_labels":
                                               tf.FixedLenFeature([FLAGS.max_predictions_per_seq_ar], tf.int64),
                                           "adr_recog_weights":
                                               tf.FixedLenFeature([FLAGS.max_predictions_per_seq_ar], tf.float32),

                                           "masked_sr_positions":
                                               tf.FixedLenFeature([FLAGS.max_predictions_per_seq_sr], tf.int64),
                                           "masked_sr_labels":
                                               tf.FixedLenFeature([FLAGS.max_predictions_per_seq_sr], tf.int64),
                                           "masked_sr_weights":
                                               tf.FixedLenFeature([FLAGS.max_predictions_per_seq_sr], tf.float32),

                                           "pointer_cd_positions_spk1":
                                               tf.FixedLenFeature([FLAGS.max_predictions_per_seq_cd], tf.int64),
                                           "pointer_cd_positions_adr1":
                                               tf.FixedLenFeature([FLAGS.max_predictions_per_seq_cd], tf.int64),
                                           "pointer_cd_positions_spk2":
                                               tf.FixedLenFeature([FLAGS.max_predictions_per_seq_cd], tf.int64),
                                           "pointer_cd_positions_adr2":
                                               tf.FixedLenFeature([FLAGS.max_predictions_per_seq_cd], tf.int64),
                                           "pointer_cd_positions_spk3":
                                               tf.FixedLenFeature([FLAGS.max_predictions_per_seq_cd], tf.int64),
                                           "pointer_cd_positions_adr3":
                                               tf.FixedLenFeature([FLAGS.max_predictions_per_seq_cd], tf.int64),
                                           "pointer_cd_weights":
                                               tf.FixedLenFeature([FLAGS.max_predictions_per_seq_cd], tf.float32),

                                           "input_ids_msur":
                                               tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
                                           "input_mask_msur":
                                               tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
                                           "segment_ids_msur":
                                               tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
                                           "speaker_ids_msur":
                                               tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
                                           "masked_sur_positions":
                                               tf.FixedLenFeature([FLAGS.max_utr_length], tf.int64),
                                           "masked_sur_ids":
                                               tf.FixedLenFeature([FLAGS.max_utr_length], tf.int64),
                                           "masked_sur_weights":
                                               tf.FixedLenFeature([FLAGS.max_utr_length], tf.float32),

                                           "input_ids_snd":
                                               tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
                                           "input_mask_snd":
                                               tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
                                           "segment_ids_snd":
                                               tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
                                           "speaker_ids_snd":
                                               tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
                                           "next_thread_labels":
                                               tf.FixedLenFeature([1], tf.int64),
                                       }
                                       )
    # So cast all int64 to int32.
    for name in list(input_data.keys()):
        t = input_data[name]
        if t.dtype == tf.int64:
            t = tf.to_int32(t)
        input_data[name] = t

    input_ids_mlm_nsp = input_data["input_ids_mlm_nsp"]
    input_mask_mlm_nsp = input_data["input_mask_mlm_nsp"]
    segment_ids_mlm_nsp = input_data["segment_ids_mlm_nsp"]
    speaker_ids_mlm_nsp = input_data["speaker_ids_mlm_nsp"]
    next_sentence_labels = input_data["next_sentence_labels"]
    masked_lm_positions = input_data["masked_lm_positions"]
    masked_lm_ids = input_data["masked_lm_ids"]
    masked_lm_weights = input_data["masked_lm_weights"]

    cls_positions = input_data["cls_positions"]
    input_ids_ar_msr_pcd = input_data["input_ids_ar_msr_pcd"]
    input_mask_ar_msr_pcd = input_data["input_mask_ar_msr_pcd"]
    segment_ids_ar_msr_pcd = input_data["segment_ids_ar_msr_pcd"]
    speaker_ids_ar_msr_pcd = input_data["speaker_ids_ar_msr_pcd"]

    adr_recog_positions = input_data["adr_recog_positions"]
    adr_recog_labels = input_data["adr_recog_labels"]
    adr_recog_weights = input_data["adr_recog_weights"]

    masked_sr_positions = input_data["masked_sr_positions"]
    masked_sr_labels = input_data["masked_sr_labels"]
    masked_sr_weights = input_data["masked_sr_weights"]

    pointer_cd_positions_spk1 = input_data["pointer_cd_positions_spk1"]
    pointer_cd_positions_adr1 = input_data["pointer_cd_positions_adr1"]
    pointer_cd_positions_spk2 = input_data["pointer_cd_positions_spk2"]
    pointer_cd_positions_adr2 = input_data["pointer_cd_positions_adr2"]
    pointer_cd_positions_spk3 = input_data["pointer_cd_positions_spk3"]
    pointer_cd_positions_adr3 = input_data["pointer_cd_positions_adr3"]
    pointer_cd_weights = input_data["pointer_cd_weights"]

    input_ids_msur = input_data["input_ids_msur"]
    input_mask_msur = input_data["input_mask_msur"]
    segment_ids_msur = input_data["segment_ids_msur"]
    speaker_ids_msur = input_data["speaker_ids_msur"]
    masked_sur_positions = input_data["masked_sur_positions"]
    masked_sur_ids = input_data["masked_sur_ids"]
    masked_sur_weights = input_data["masked_sur_weights"]

    input_ids_snd = input_data["input_ids_snd"]
    input_mask_snd = input_data["input_mask_snd"]
    segment_ids_snd = input_data["segment_ids_snd"]
    speaker_ids_snd = input_data["speaker_ids_snd"]
    next_thread_labels = input_data["next_thread_labels"]

    return input_ids_mlm_nsp, input_mask_mlm_nsp, segment_ids_mlm_nsp, speaker_ids_mlm_nsp, \
              next_sentence_labels, masked_lm_positions, masked_lm_ids, masked_lm_weights, \
              cls_positions, input_ids_ar_msr_pcd, input_mask_ar_msr_pcd, segment_ids_ar_msr_pcd, speaker_ids_ar_msr_pcd, \
              adr_recog_positions, adr_recog_labels, adr_recog_weights, masked_sr_positions, masked_sr_labels, masked_sr_weights, \
              pointer_cd_positions_spk1, pointer_cd_positions_adr1, pointer_cd_positions_spk2, pointer_cd_positions_adr2, \
              pointer_cd_positions_spk3, pointer_cd_positions_adr3, pointer_cd_weights, \
              input_ids_msur, input_mask_msur, segment_ids_msur, speaker_ids_msur, masked_sur_positions, masked_sur_ids, masked_sur_weights, \
              input_ids_snd, input_mask_snd, segment_ids_snd, speaker_ids_snd, next_thread_labels


def model_fn_builder(features, is_training, bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu, use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    input_ids_mlm_nsp, input_mask_mlm_nsp, segment_ids_mlm_nsp, speaker_ids_mlm_nsp, \
        next_sentence_labels, masked_lm_positions, masked_lm_ids, masked_lm_weights, \
        cls_positions, input_ids_ar_msr_pcd, input_mask_ar_msr_pcd, segment_ids_ar_msr_pcd, speaker_ids_ar_msr_pcd, \
        adr_recog_positions, adr_recog_labels, adr_recog_weights, masked_sr_positions, masked_sr_labels, masked_sr_weights, \
        pointer_cd_positions_spk1, pointer_cd_positions_adr1, pointer_cd_positions_spk2, pointer_cd_positions_adr2, \
        pointer_cd_positions_spk3, pointer_cd_positions_adr3, pointer_cd_weights, \
        input_ids_msur, input_mask_msur, segment_ids_msur, speaker_ids_msur, masked_sur_positions, masked_sur_ids, masked_sur_weights, \
        input_ids_snd, input_mask_snd, segment_ids_snd, speaker_ids_snd, next_thread_labels = features

    model_mlm_nsp = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids_mlm_nsp,
        input_mask=input_mask_mlm_nsp,
        token_type_ids=segment_ids_mlm_nsp,
        speaker_ids=speaker_ids_mlm_nsp,
        use_one_hot_embeddings=use_one_hot_embeddings,
        scope="bert",
        scope_reuse=False)

    (masked_lm_loss, masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
         bert_config, model_mlm_nsp.get_sequence_output(), model_mlm_nsp.get_embedding_table(), masked_lm_positions, masked_lm_ids, masked_lm_weights)

    (next_sentence_loss, next_sentence_example_loss, next_sentence_log_probs) = get_next_sentence_output(
         bert_config, model_mlm_nsp.get_pooled_output(), next_sentence_labels)

    model_ar_msr_pcd = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids_ar_msr_pcd,
        input_mask=input_mask_ar_msr_pcd,
        token_type_ids=segment_ids_ar_msr_pcd,
        speaker_ids=speaker_ids_ar_msr_pcd,
        use_one_hot_embeddings=use_one_hot_embeddings,
        scope="bert",
        scope_reuse=True)
    
    (adr_recog_loss, adr_recog_example_loss, adr_recog_log_probs) = get_replyto_utterance_recognition_output(
         bert_config, model_ar_msr_pcd.get_sequence_output(), cls_positions, adr_recog_positions, adr_recog_labels, adr_recog_weights)

    (masked_sr_loss, masked_sr_example_loss, masked_sr_log_probs) = get_identical_speaker_searching_output(
         bert_config, model_ar_msr_pcd.get_sequence_output(), cls_positions, masked_sr_positions, masked_sr_labels, masked_sr_weights)

    (pointer_cd_loss, pointer_cd_example_loss, pointer_cd_log_probs) = get_pointer_consistency_distinction_output(
         bert_config, model_ar_msr_pcd.get_sequence_output(), cls_positions, pointer_cd_positions_spk1, pointer_cd_positions_adr1, 
         pointer_cd_positions_spk2, pointer_cd_positions_adr2, pointer_cd_positions_spk3, pointer_cd_positions_adr3, pointer_cd_weights)

    model_msur = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids_msur,
        input_mask=input_mask_msur,
        token_type_ids=segment_ids_msur,
        speaker_ids=speaker_ids_msur,
        use_one_hot_embeddings=use_one_hot_embeddings,
        scope="bert",
        scope_reuse=True)

    (masked_sur_loss, masked_sur_example_loss, masked_sur_log_probs) = get_masked_shared_utterance_restoration_output(
         bert_config, model_msur.get_sequence_output(), model_msur.get_embedding_table(), masked_sur_positions, masked_sur_ids, masked_sur_weights)

    model_snd = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids_snd,
        input_mask=input_mask_snd,
        token_type_ids=segment_ids_snd,
        speaker_ids=speaker_ids_snd,
        use_one_hot_embeddings=use_one_hot_embeddings,
        scope="bert",
        scope_reuse=True)

    (shared_nd_loss, shared_nd_example_loss, shared_nd_log_probs) = get_shared_node_detection_output(
         bert_config, model_snd.get_pooled_output(), next_thread_labels)
    
    total_loss = masked_lm_loss + next_sentence_loss + \
                    adr_recog_loss + masked_sr_loss + pointer_cd_loss + \
                    masked_sur_loss + shared_nd_loss

    tvars = tf.trainable_variables()

    if init_checkpoint:
        (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
        init_string = ""
        if var.name in initialized_variable_names:
            init_string = ", *INIT_FROM_CKPT*"
        tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

    train_op = optimization.create_optimizer(total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

    metrics = metric_fn(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids, masked_lm_weights, 
                            next_sentence_example_loss, next_sentence_log_probs, next_sentence_labels, 
                            adr_recog_example_loss, adr_recog_log_probs, adr_recog_labels, adr_recog_weights,
                            masked_sr_example_loss, masked_sr_log_probs, masked_sr_labels, masked_sr_weights, 
                            pointer_cd_example_loss, pointer_cd_log_probs, pointer_cd_weights,
                            masked_sur_example_loss, masked_sur_log_probs, masked_sur_ids, masked_sur_weights,
                            shared_nd_example_loss, shared_nd_log_probs, next_thread_labels)

    return train_op, total_loss, metrics, input_ids_mlm_nsp


def get_masked_lm_output(bert_config, input_tensor, output_weights, positions, label_ids, label_weights):
  """Get loss and log probs for the masked LM."""
  input_tensor = gather_indexes(input_tensor, positions)  # [batch_size*max_predictions_per_seq, dim]

  with tf.variable_scope("cls/predictions"):
    # We apply one more non-linear transformation before the output layer.
    with tf.variable_scope("transform"):
      input_tensor = tf.layers.dense(
          input_tensor,
          units=bert_config.hidden_size,
          activation=modeling.get_activation(bert_config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              bert_config.initializer_range))
      input_tensor = modeling.layer_norm(input_tensor)

    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.
    output_bias = tf.get_variable(
        "output_bias",
        shape=[bert_config.vocab_size],
        initializer=tf.zeros_initializer())
    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)   # [batch_size*max_predictions_per_seq, vocab_size]

    label_ids = tf.reshape(label_ids, [-1])
    label_weights = tf.reshape(label_weights, [-1])
    one_hot_labels = tf.one_hot(label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

    # The `label_weights` tensor has a value of 1.0 for every real prediction and 0.0 for the padding predictions.
    per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])  # [batch_size*max_predictions_per_seq, ]
    numerator = tf.reduce_sum(label_weights * per_example_loss)               # [1, ]
    denominator = tf.reduce_sum(label_weights) + 1e-5
    loss = numerator / denominator

  return (loss, per_example_loss, log_probs)


def get_next_sentence_output(bert_config, input_tensor, labels):
  """Get loss and log probs for the next sentence prediction."""

  # Simple binary classification. Note that 0 is "next sentence" and 1 is
  # "random sentence". This weight matrix is not used after pre-training.
  with tf.variable_scope("cls/seq_relationship"):
    output_weights = tf.get_variable(
        "output_weights",
        shape=[2, bert_config.hidden_size],
        initializer=modeling.create_initializer(bert_config.initializer_range))
    output_bias = tf.get_variable(
        "output_bias", shape=[2], initializer=tf.zeros_initializer())

    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)                           # [batch_size, 2]
    labels = tf.reshape(labels, [-1])
    one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
    per_example_loss = - tf.reduce_sum(one_hot_labels * log_probs, axis=-1)  # [batch_size, ]
    loss = tf.reduce_mean(per_example_loss)                                  # [1, ]

    return (loss, per_example_loss, log_probs)


def get_replyto_utterance_recognition_output(bert_config, input_tensor, cls_positions, adr_recog_positions, adr_recog_labels, adr_recog_weights):
  """Get loss and log probs for the Reply-to Utterance Recognition (RUR)."""
  # input_tensor:        [batch_size, max_sequence_len, dim]
  # cls_positions:       [batch_size, max_utr_num]
  # adr_recog_positions: [batch_size, max_predictions_per_seq_ar]
  # adr_recog_labels:    [batch_size, max_predictions_per_seq_ar]
  # adr_recog_weights:   [batch_size, max_predictions_per_seq_ar]
  
  cls_shape = modeling.get_shape_list(cls_positions, expected_rank=2)
  max_utr_num = cls_shape[1]

  input_tensor = gather_indexes(input_tensor, cls_positions)           # [batch_size*max_utr_num, dim]
  input_shape = modeling.get_shape_list(input_tensor, expected_rank=2)
  width = input_shape[1]

  with tf.variable_scope("cls/addressee_recognize"):
    # We apply one more non-linear transformation before the output layer.
    with tf.variable_scope("transform"):
      input_tensor = tf.layers.dense(
          input_tensor,
          units=bert_config.hidden_size,
          activation=modeling.get_activation(bert_config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              bert_config.initializer_range))
      input_tensor = modeling.layer_norm(input_tensor)                 # [batch_size*max_utr_num, dim]

    input_tensor = tf.reshape(input_tensor, [-1, max_utr_num, width])  # [batch_size, max_utr_num, dim]
    output_weights = tf.get_variable(
        "output_weights",
        shape=[width, width],
        initializer=modeling.create_initializer(bert_config.initializer_range))
    logits = tf.matmul(tf.einsum('aij,jk->aik', input_tensor, output_weights),
                       input_tensor, transpose_b=True)                 # [batch_size, max_utr_num, max_utr_num]

    # mask = [[0. 0. 0. 0. 0.]
    #         [1. 0. 0. 0. 0.]
    #         [1. 1. 0. 0. 0.]
    #         [1. 1. 1. 0. 0.]
    #         [1. 1. 1. 1. 0.]]
    mask = tf.matrix_band_part(tf.ones((max_utr_num, max_utr_num)), -1, 0) - tf.matrix_band_part(tf.ones((max_utr_num, max_utr_num)), 0, 0)
    logits = logits * mask + -1e9 * (1-mask)              # [batch_size, max_utr_num, max_utr_num]

    logits = gather_indexes(logits, adr_recog_positions)  # [batch_size*max_predictions_per_seq_ar, max_utr_num]
    log_probs = tf.nn.log_softmax(logits, axis=-1)        # [batch_size*max_predictions_per_seq_ar, max_utr_num]

    label_ids = tf.reshape(adr_recog_labels, [-1])        # [batch_size*max_predictions_per_seq_ar, ]
    label_weights = tf.reshape(adr_recog_weights, [-1])   # [batch_size*max_predictions_per_seq_ar, ]
    one_hot_labels = tf.one_hot(label_ids, depth=max_utr_num, dtype=tf.float32)  # [batch_size*max_predictions_per_seq_ar, max_utr_num]

    # The `label_weights` tensor has a value of 1.0 for every real prediction and 0.0 for the padding predictions.
    per_example_loss = - tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])  # [batch_size*max_predictions_per_seq_ar, ]
    numerator = tf.reduce_sum(label_weights * per_example_loss)                # [1, ]
    denominator = tf.reduce_sum(label_weights) + 1e-5                          # [1, ]
    loss = numerator / denominator

    return (loss, per_example_loss, log_probs)


def get_identical_speaker_searching_output(bert_config, input_tensor, cls_positions, masked_sr_positions, masked_sr_labels, masked_sr_weights):
  """Get loss and log probs for the Identical Speaker Searching (ISS)."""
  # input_tensor:        [batch_size, max_sequence_len, dim]
  # cls_positions:       [batch_size, max_utr_num]
  # masked_sr_positions: [batch_size, max_predictions_per_seq_sr]
  # masked_sr_labels:    [batch_size, max_predictions_per_seq_sr]
  # masked_sr_weights:   [batch_size, max_predictions_per_seq_sr]
  
  cls_shape = modeling.get_shape_list(cls_positions, expected_rank=2)
  max_utr_num = cls_shape[1]

  input_tensor = gather_indexes(input_tensor, cls_positions)  # [batch_size*max_utr_num, dim]
  input_shape = modeling.get_shape_list(input_tensor, expected_rank=2)
  width = input_shape[1]

  with tf.variable_scope("cls/speaker_restore"):
    # We apply one more non-linear transformation before the output layer.
    with tf.variable_scope("transform"):
      input_tensor = tf.layers.dense(
          input_tensor,
          units=bert_config.hidden_size,
          activation=modeling.get_activation(bert_config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              bert_config.initializer_range))
      input_tensor = modeling.layer_norm(input_tensor)  # [batch_size*max_utr_num, dim]

    input_tensor = tf.reshape(input_tensor, [-1, max_utr_num, width])  # [batch_size, max_utr_num, dim]
    output_weights = tf.get_variable(
        "output_weights",
        shape=[width, width],
        initializer=modeling.create_initializer(bert_config.initializer_range))
    logits = tf.matmul(tf.einsum('aij,jk->aik', input_tensor, output_weights),
                       input_tensor, transpose_b=True)                 # [batch_size, max_utr_num, max_utr_num]

    # mask = [[0. 0. 0. 0. 0.]
    #         [1. 0. 0. 0. 0.]
    #         [1. 1. 0. 0. 0.]
    #         [1. 1. 1. 0. 0.]
    #         [1. 1. 1. 1. 0.]]
    mask = tf.matrix_band_part(tf.ones((max_utr_num, max_utr_num)), -1, 0) - tf.matrix_band_part(tf.ones((max_utr_num, max_utr_num)), 0, 0)
    logits = logits * mask + -1e9 * (1-mask)              # [batch_size, max_utr_num, max_utr_num]

    logits = gather_indexes(logits, masked_sr_positions)  # [batch_size*max_predictions_per_seq_sr, max_utr_num]
    log_probs = tf.nn.log_softmax(logits, axis=-1)        # [batch_size*max_predictions_per_seq_sr, max_utr_num]

    label_ids = tf.reshape(masked_sr_labels, [-1])        # [batch_size*max_predictions_per_seq_sr, ]
    label_weights = tf.reshape(masked_sr_weights, [-1])   # [batch_size*max_predictions_per_seq_sr, ]
    one_hot_labels = tf.one_hot(label_ids, depth=max_utr_num, dtype=tf.float32)  # [batch_size*max_predictions_per_seq_sr, max_utr_num]

    # The `label_weights` tensor has a value of 1.0 for every real prediction and 0.0 for the padding predictions.
    per_example_loss = - tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])  # [batch_size*max_predictions_per_seq_sr, ]
    numerator = tf.reduce_sum(label_weights * per_example_loss)                # [1, ]
    denominator = tf.reduce_sum(label_weights) + 1e-5                          # [1, ]
    loss = numerator / denominator

    return (loss, per_example_loss, log_probs)


def get_pointer_consistency_distinction_output(bert_config, input_tensor, cls_positions, pointer_cd_positions_spk1, pointer_cd_positions_adr1, 
        pointer_cd_positions_spk2, pointer_cd_positions_adr2, pointer_cd_positions_spk3, pointer_cd_positions_adr3, pointer_cd_weights):
  """Get loss and log probs for the Pointer Consistency Distinction (PCD)."""
  # input_tensor:              [batch_size, max_sequence_len, dim]
  # cls_positions:             [batch_size, max_utr_num]
  # pointer_cd_positions_spk1: [batch_size, max_predictions_per_seq_cd]
  
  cls_shape = modeling.get_shape_list(cls_positions, expected_rank=2)
  max_utr_num = cls_shape[1]

  positions_shape = modeling.get_shape_list(pointer_cd_positions_spk1, expected_rank=2)
  max_predictions_per_seq_cd = positions_shape[1]

  input_tensor = gather_indexes(input_tensor, cls_positions)  # [batch_size*max_utr_num, dim]
  input_shape = modeling.get_shape_list(input_tensor, expected_rank=2)
  width = input_shape[1]

  with tf.variable_scope("cls/pointer_consistency_distinct"):
    # We apply one more non-linear transformation before the output layer.
    # This matrix is not used after pre-training.
    with tf.variable_scope("transform"):
      input_tensor = tf.layers.dense(
          input_tensor,
          units=bert_config.hidden_size,
          activation=modeling.get_activation(bert_config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              bert_config.initializer_range))
      input_tensor = modeling.layer_norm(input_tensor)  # [batch_size*max_utr_num, dim]

    input_tensor = tf.reshape(input_tensor, [-1, max_utr_num, width])  # [batch_size, max_utr_num, dim]
    spk1 = gather_indexes(input_tensor, pointer_cd_positions_spk1)     # [batch_size*max_predictions_per_seq_cd, dim]
    adr1 = gather_indexes(input_tensor, pointer_cd_positions_adr1)     # [batch_size*max_predictions_per_seq_cd, dim]
    spk2 = gather_indexes(input_tensor, pointer_cd_positions_spk2)     # [batch_size*max_predictions_per_seq_cd, dim]
    adr2 = gather_indexes(input_tensor, pointer_cd_positions_adr2)     # [batch_size*max_predictions_per_seq_cd, dim]
    spk3 = gather_indexes(input_tensor, pointer_cd_positions_spk3)     # [batch_size*max_predictions_per_seq_cd, dim]
    adr3 = gather_indexes(input_tensor, pointer_cd_positions_adr3)     # [batch_size*max_predictions_per_seq_cd, dim]

    # [multiply, minus] + FFNN
    W_aggre = tf.get_variable(
        "W_aggre",
        shape=[width*2, width],
        initializer=modeling.create_initializer(bert_config.initializer_range))
    b_aggre = tf.get_variable(
        "b_aggre", 
        shape=[width, ], 
        initializer=modeling.create_initializer(bert_config.initializer_range))

    pointer1 = tf.concat(values=[tf.multiply(spk1, adr1), spk1 - adr1], axis=-1)
    pointer1 = tf.nn.relu(tf.matmul(pointer1, W_aggre) + b_aggre)
    
    pointer2 = tf.concat(values=[tf.multiply(spk2, adr2), spk2 - adr2], axis=-1)
    pointer2 = tf.nn.relu(tf.matmul(pointer2, W_aggre) + b_aggre)
    
    pointer3 = tf.concat(values=[tf.multiply(spk3, adr3), spk3 - adr3], axis=-1)
    pointer3 = tf.nn.relu(tf.matmul(pointer3, W_aggre) + b_aggre)

    output_weights = tf.get_variable(
        "output_weights",
        shape=[width, width],
        initializer=modeling.create_initializer(bert_config.initializer_range))
    logits12 = tf.reduce_sum(tf.multiply(tf.matmul(pointer1, output_weights), pointer2), axis=[-1])  # [batch_size*max_predictions_per_seq_cd, ]
    log_probs12 = tf.sigmoid(logits12, name="logits12")
    logits13 = tf.reduce_sum(tf.multiply(tf.matmul(pointer1, output_weights), pointer3), axis=[-1])  # [batch_size*max_predictions_per_seq_cd, ]
    log_probs13 = tf.sigmoid(logits13, name="logits13")
    logits23 = tf.reduce_sum(tf.multiply(tf.matmul(pointer2, output_weights), pointer3), axis=[-1])  # [batch_size*max_predictions_per_seq_cd, ]
    log_probs23 = tf.sigmoid(logits23, name="logits23")

    label_weights = tf.reshape(pointer_cd_weights, [-1])                                             # [batch_size*max_predictions_per_seq_cd, ]
    
    # The `label_weights` tensor has a value of 1.0 for every real prediction and 0.0 for the padding predictions.
    delta = 0.4
    per_example_loss = tf.maximum(0.0, delta - log_probs12 + log_probs13)
    numerator = tf.reduce_sum(label_weights * per_example_loss)        # [1, ]
    denominator = tf.reduce_sum(label_weights) + 1e-5                  # [1, ]
    loss = numerator / denominator

    return (loss, per_example_loss, log_probs12 - log_probs13)


def get_masked_shared_utterance_restoration_output(bert_config, input_tensor, output_weights, positions, label_ids, label_weights):
  """Get loss and log probs for the Masked Shared Utterance Restoration (MSUR)."""

  input_tensor = gather_indexes(input_tensor, positions)  # [batch_size*max_utr_length, dim]

  with tf.variable_scope("cls/shared_utterance_restore"):
    # We apply one more non-linear transformation before the output layer.
    with tf.variable_scope("transform"):
      input_tensor = tf.layers.dense(
          input_tensor,
          units=bert_config.hidden_size,
          activation=modeling.get_activation(bert_config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              bert_config.initializer_range))
      input_tensor = modeling.layer_norm(input_tensor)

    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.
    output_bias = tf.get_variable(
        "output_bias",
        shape=[bert_config.vocab_size],
        initializer=tf.zeros_initializer())
    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)   # [batch_size*max_utr_length, vocab_size]

    label_ids = tf.reshape(label_ids, [-1])
    label_weights = tf.reshape(label_weights, [-1])
    one_hot_labels = tf.one_hot(label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

    # The `label_weights` tensor has a value of 1.0 for every real prediction and 0.0 for the padding predictions.
    per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])  # [batch_size*max_predictions_per_seq, ]
    numerator = tf.reduce_sum(label_weights * per_example_loss)               # [1, ]
    denominator = tf.reduce_sum(label_weights) + 1e-5
    loss = numerator / denominator

  return (loss, per_example_loss, log_probs)


def get_shared_node_detection_output(bert_config, input_tensor, labels):

  with tf.variable_scope("cls/shared_node_detect"):
    output_weights = tf.get_variable(
        "output_weights",
        shape=[2, bert_config.hidden_size],
        initializer=modeling.create_initializer(bert_config.initializer_range))
    output_bias = tf.get_variable(
        "output_bias", shape=[2], initializer=tf.zeros_initializer())

    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)  # [batch_size, 2]
    labels = tf.reshape(labels, [-1])
    one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
    per_example_loss = - tf.reduce_sum(one_hot_labels * log_probs, axis=-1)  # [batch_size, ]
    loss = tf.reduce_mean(per_example_loss)                                  # [1, ]

    return (loss, per_example_loss, log_probs)


def gather_indexes(sequence_tensor, positions):
  """Gathers the vectors at the specific positions over a minibatch."""
  # sequence_tensor = [batch_size, seq_length, width]
  # positions = [batch_size, max_predictions_per_seq]

  sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
  batch_size = sequence_shape[0]
  seq_length = sequence_shape[1]
  width = sequence_shape[2]

  flat_offsets = tf.reshape(tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])  # [batch_size, 1]
  flat_positions = tf.reshape(positions + flat_offsets, [-1])         # [batch_size*max_predictions_per_seq, ]
  flat_sequence_tensor = tf.reshape(sequence_tensor, [batch_size * seq_length, width])
  output_tensor = tf.gather(flat_sequence_tensor, flat_positions)     # [batch_size*max_predictions_per_seq, width]
  return output_tensor


def metric_fn(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids, masked_lm_weights, 
                  next_sentence_example_loss, next_sentence_log_probs, next_sentence_labels,
                  adr_recog_example_loss, adr_recog_log_probs, adr_recog_labels, adr_recog_weights,
                  masked_sr_example_loss, masked_sr_log_probs, masked_sr_labels, masked_sr_weights,
                  pointer_cd_example_loss, pointer_cd_log_probs, pointer_cd_weights, 
                  masked_sur_example_loss, masked_sur_log_probs, masked_sur_ids, masked_sur_weights,
                  shared_nd_example_loss, shared_nd_log_probs, next_thread_labels):
  """Computes the loss and accuracy of the model."""

  # adr_recog_example_loss: [batch_size*max_predictions_per_seq_ar, ]
  # adr_recog_log_probs:    [batch_size*max_predictions_per_seq_ar, max_utr_num]
  # adr_recog_labels:       [batch_size, max_predictions_per_seq_ar]
  # adr_recog_weights:      [batch_size, max_predictions_per_seq_ar]

  # masked_sr_example_loss: [batch_size*max_utr_num, ]
  # masked_sr_log_probs:    [batch_size*max_utr_num, max_utr_num]
  # masked_sr_labels:       [batch_size, max_utr_num]
  # masked_sr_weights:      [batch_size, max_utr_num]
  
  # pointer_cd_example_loss: [batch_size*max_predictions_per_seq_cd, ]
  # pointer_cd_log_probs:    [batch_size*max_predictions_per_seq_cd, ]
  # pointer_cd_weights:      [batch_size, max_predictions_per_seq_cd]
  
  # masked_sur_example_loss: [batch_size*max_utr_length, ]
  # masked_sur_log_probs:    [batch_size*max_utr_length, vocab_size]
  # masked_sur_labels:       [batch_size, max_utr_length]
  # masked_sur_weights:      [batch_size, max_utr_length]
  
  # shared_nd_example_loss: [batch_size, ]
  # shared_nd_log_probs:    [batch_size, 2]
  # next_thread_labels:     [batch_size, ]
  
  masked_lm_predictions = tf.argmax(masked_lm_log_probs, axis=-1, output_type=tf.int32)         # [batch_size*max_predictions_per_seq, ]
  masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
  masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
  masked_lm_accuracy = tf.metrics.accuracy(
      labels=masked_lm_ids,
      predictions=masked_lm_predictions,
      weights=masked_lm_weights)
  masked_lm_mean_loss = tf.metrics.mean(
      values=masked_lm_example_loss, 
      weights=masked_lm_weights)

  next_sentence_predictions = tf.argmax(next_sentence_log_probs, axis=-1, output_type=tf.int32) # [batch_size, ]
  next_sentence_accuracy = tf.metrics.accuracy(
      labels=next_sentence_labels, 
      predictions=next_sentence_predictions)
  next_sentence_mean_loss = tf.metrics.mean(
      values=next_sentence_example_loss)
  
  adr_recog_predictions = tf.argmax(adr_recog_log_probs, axis=-1, output_type=tf.int32)  # [batch_size*max_predictions_per_seq_ar, ]
  adr_recog_labels = tf.reshape(adr_recog_labels, [-1])                                  # [batch_size*max_predictions_per_seq_ar, ]
  adr_recog_weights = tf.reshape(adr_recog_weights, [-1])                                # [batch_size*max_predictions_per_seq_ar, ]
  adr_recog_accuracy = tf.metrics.accuracy(
      labels=adr_recog_labels,
      predictions=adr_recog_predictions,
      weights=adr_recog_weights)
  adr_recog_mean_loss = tf.metrics.mean(
      values=adr_recog_example_loss, 
      weights=adr_recog_weights)

  masked_sr_predictions = tf.argmax(masked_sr_log_probs, axis=-1, output_type=tf.int32)  # [batch_size*max_predictions_per_seq_sr, ]
  masked_sr_labels = tf.reshape(masked_sr_labels, [-1])                                  # [batch_size*max_predictions_per_seq_sr, ]
  masked_sr_weights = tf.reshape(masked_sr_weights, [-1])                                # [batch_size*max_predictions_per_seq_sr, ]
  masked_sr_accuracy = tf.metrics.accuracy(
      labels=masked_sr_labels,
      predictions=masked_sr_predictions,
      weights=masked_sr_weights)
  masked_sr_mean_loss = tf.metrics.mean(
      values=masked_sr_example_loss, 
      weights=masked_sr_weights)

  pointer_cd_weights = tf.reshape(pointer_cd_weights, [-1])  # [batch_size*max_predictions_per_seq_cd, ]
  pointer_cd_mean_simi = tf.metrics.mean(
      values=pointer_cd_log_probs, 
      weights=pointer_cd_weights)
  pointer_cd_mean_loss = tf.metrics.mean(
      values=pointer_cd_example_loss, 
      weights=pointer_cd_weights)

  masked_sur_predictions = tf.argmax(masked_sur_log_probs, axis=-1, output_type=tf.int32) # [batch_size*max_utr_length, ]
  masked_sur_ids = tf.reshape(masked_sur_ids, [-1])                                       # [batch_size*max_utr_length, ]
  masked_sur_weights = tf.reshape(masked_sur_weights, [-1])                               # [batch_size*max_utr_length, ]
  masked_sur_accuracy = tf.metrics.accuracy(
      labels=masked_sur_ids,
      predictions=masked_sur_predictions,
      weights=masked_sur_weights)
  masked_sur_mean_loss = tf.metrics.mean(
      values=masked_sur_example_loss, 
      weights=masked_sur_weights)

  shared_nd_predictions = tf.argmax(shared_nd_log_probs, axis=-1, output_type=tf.int32)   # [batch_size, ]
  shared_nd_accuracy = tf.metrics.accuracy(
      labels=next_thread_labels, 
      predictions=shared_nd_predictions)
  shared_nd_mean_loss = tf.metrics.mean(
      values=shared_nd_example_loss)

  return {
      "masked_lm_accuracy": masked_lm_accuracy,
      "masked_lm_loss": masked_lm_mean_loss,
      "next_sentence_accuracy": next_sentence_accuracy,
      "next_sentence_loss": next_sentence_mean_loss,
      "adr_recog_accuracy": adr_recog_accuracy,
      "adr_recog_loss": adr_recog_mean_loss,
      "masked_sr_accuracy": masked_sr_accuracy,
      "masked_sr_loss": masked_sr_mean_loss,
      "pointer_cd_simi": pointer_cd_mean_simi,
      "pointer_cd_loss": pointer_cd_mean_loss,
      "masked_sur_accuracy": masked_sur_accuracy,
      "masked_sur_loss": masked_sur_mean_loss,
      "shared_nd_accuracy": shared_nd_accuracy,
      "shared_nd_loss": shared_nd_mean_loss
  }


def run_epoch(epoch, sess, saver, output_dir, epoch_save_step, mid_save_step, 
                input_ids, eval_metrics, total_loss, train_op, eval_op):

    total_sample = 0
    # accumulate_loss = 0
    step = 0
    t0 = time()

    tf.logging.info("*** Start epoch {} training ***".format(epoch))
    try:
        while True:
            step += 1
            _input_ids, batch_metrics, batch_loss, _, _ = sess.run([input_ids, eval_metrics, total_loss, train_op, eval_op] )
            masked_lm_accuracy, masked_lm_loss, next_sentence_accuracy, next_sentence_loss, \
                adr_recog_accuracy, adr_recog_loss, masked_sr_accuracy, masked_sr_loss, \
                pointer_cd_simi, pointer_cd_loss, masked_sur_accuracy, masked_sur_loss, \
                shared_nd_accuracy, shared_nd_loss = batch_metrics

            batch_sample = len(_input_ids)
            total_sample += batch_sample
            # accumulate_loss += batch_loss * batch_sample

            # print
            print_every_step = 200
            if step % print_every_step == 0:
                tf.logging.info("Step: {}, Loss: {:.4f}, Sample: {}, Time (min): {:.2f}".format(
                       step, batch_loss, total_sample, (time()-t0)/60))
                tf.logging.info('MLM_accuracy: {:.6f}, MLM_loss: {:.6f}, NSP_accuracy: {:.6f}, NSP_loss: {:.6f}, '
                                'RUR_accuracy: {:.6f}, RUR_loss: {:.6f}, ISS_accuracy: {:.6f}, ISS_loss: {:.6f}, '
                                'PCD_similarity: {:.6f}, PCD_loss: {:.6f}, MSUR_accuracy: {:.6f}, MSUR_loss: {:.6f}, '
                                'SND_accuracy: {:.6f}, SND_loss: {:.6f}'.format(
                                  masked_lm_accuracy, masked_lm_loss, next_sentence_accuracy, next_sentence_loss, 
                                  adr_recog_accuracy, adr_recog_loss, masked_sr_accuracy, masked_sr_loss, 
                                  pointer_cd_simi, pointer_cd_loss, masked_sur_accuracy, masked_sur_loss,
                                  shared_nd_accuracy, shared_nd_loss))

            if (step % mid_save_step == 0) or (step % epoch_save_step == 0):
                # c_time = str(int(time()))
                save_path = os.path.join(output_dir, 'pretrained_bert_model_epoch_{}_step_{}'.format(epoch, step))
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                saver.save(sess, os.path.join(save_path, 'bert_model_epoch_{}_step_{}.ckpt'.format(epoch, step)), global_step=step)
                tf.logging.info('========== Save model at epoch: {}, step: {} =========='.format(epoch, step))
                tf.logging.info("Step: {}, Loss: {:.4f}, Sample: {}, Time (min): {:.2f}".format(
                       step, batch_loss, total_sample, (time()-t0)/60))
                tf.logging.info('MLM_accuracy: {:.6f}, MLM_loss: {:.6f}, NSP_accuracy: {:.6f}, NSP_loss: {:.6f}, '
                                'RUR_accuracy: {:.6f}, RUR_loss: {:.6f}, ISS_accuracy: {:.6f}, ISS_loss: {:.6f}, '
                                'PCD_similarity: {:.6f}, PCD_loss: {:.6f}, MSUR_accuracy: {:.6f}, MSUR_loss: {:.6f}, '
                                'SND_accuracy: {:.6f}, SND_loss: {:.6f}'.format(
                                  masked_lm_accuracy, masked_lm_loss, next_sentence_accuracy, next_sentence_loss, 
                                  adr_recog_accuracy, adr_recog_loss, masked_sr_accuracy, masked_sr_loss, 
                                  pointer_cd_simi, pointer_cd_loss, masked_sur_accuracy, masked_sur_loss,
                                  shared_nd_accuracy, shared_nd_loss))
                
    except tf.errors.OutOfRangeError:
        tf.logging.info('*** Epoch {} is finished ***'.format(epoch))
        pass


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    print_configuration_op(FLAGS)

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    train_data_size = count_data_size(FLAGS.input_file)
    tf.logging.info('*** train data size: {} ***'.format(train_data_size))

    num_train_steps = train_data_size // FLAGS.train_batch_size * FLAGS.num_train_epochs
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
    epoch_save_step = train_data_size // FLAGS.train_batch_size

    buffer_size = 1000
    filenames = tf.placeholder(tf.string, shape=[None])
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse_exmp)  # Parse the record into tensors.
    dataset = dataset.repeat(1)
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(FLAGS.train_batch_size)
    iterator = dataset.make_initializable_iterator()

    input_ids_mlm_nsp, input_mask_mlm_nsp, segment_ids_mlm_nsp, speaker_ids_mlm_nsp, \
        next_sentence_labels, masked_lm_positions, masked_lm_ids, masked_lm_weights, \
        cls_positions, input_ids_ar_msr_pcd, input_mask_ar_msr_pcd, segment_ids_ar_msr_pcd, speaker_ids_ar_msr_pcd, \
        adr_recog_positions, adr_recog_labels, adr_recog_weights, masked_sr_positions, masked_sr_labels, masked_sr_weights, \
        pointer_cd_positions_spk1, pointer_cd_positions_adr1, pointer_cd_positions_spk2, pointer_cd_positions_adr2, \
        pointer_cd_positions_spk3, pointer_cd_positions_adr3, pointer_cd_weights, \
        input_ids_msur, input_mask_msur, segment_ids_msur, speaker_ids_msur, masked_sur_positions, masked_sur_ids, masked_sur_weights, \
        input_ids_snd, input_mask_snd, segment_ids_snd, speaker_ids_snd, next_thread_labels = iterator.get_next()
    
    features = [input_ids_mlm_nsp, input_mask_mlm_nsp, segment_ids_mlm_nsp, speaker_ids_mlm_nsp, \
                    next_sentence_labels, masked_lm_positions, masked_lm_ids, masked_lm_weights, \
                    cls_positions, input_ids_ar_msr_pcd, input_mask_ar_msr_pcd, segment_ids_ar_msr_pcd, speaker_ids_ar_msr_pcd, \
                    adr_recog_positions, adr_recog_labels, adr_recog_weights, masked_sr_positions, masked_sr_labels, masked_sr_weights, \
                    pointer_cd_positions_spk1, pointer_cd_positions_adr1, pointer_cd_positions_spk2, pointer_cd_positions_adr2, \
                    pointer_cd_positions_spk3, pointer_cd_positions_adr3, pointer_cd_weights, \
                    input_ids_msur, input_mask_msur, segment_ids_msur, speaker_ids_msur, masked_sur_positions, masked_sur_ids, masked_sur_weights, \
                    input_ids_snd, input_mask_snd, segment_ids_snd, speaker_ids_snd, next_thread_labels]

    train_op, total_loss, metrics, input_ids = model_fn_builder(
        features=features,
        is_training=True,
        bert_config=bert_config,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=False,
        use_one_hot_embeddings=False)

    masked_lm_accuracy, masked_lm_accuracy_op = metrics["masked_lm_accuracy"]
    masked_lm_loss, masked_lm_loss_op = metrics["masked_lm_loss"]
    next_sentence_accuracy, next_sentence_op = metrics["next_sentence_accuracy"]
    next_sentence_loss, next_sentence_loss_op = metrics["next_sentence_loss"]
    adr_recog_accuracy, adr_recog_accuracy_op = metrics["adr_recog_accuracy"]
    adr_recog_loss, adr_recog_loss_op = metrics["adr_recog_loss"]
    masked_sr_accuracy, masked_sr_accuracy_op = metrics["masked_sr_accuracy"]
    masked_sr_loss, masked_sr_loss_op = metrics["masked_sr_loss"]
    pointer_cd_simi, pointer_cd_simi_op = metrics["pointer_cd_simi"]
    pointer_cd_loss, pointer_cd_loss_op = metrics["pointer_cd_loss"]
    masked_sur_accuracy, masked_sur_accuracy_op = metrics["masked_sur_accuracy"]
    masked_sur_loss, masked_sur_loss_op = metrics["masked_sur_loss"]
    shared_nd_accuracy, shared_nd_accuracy_op = metrics["shared_nd_accuracy"]
    shared_nd_loss, shared_nd_loss_op = metrics["shared_nd_loss"]

    eval_metrics = [masked_lm_accuracy, masked_lm_loss, next_sentence_accuracy, next_sentence_loss, \
                        adr_recog_accuracy, adr_recog_loss, masked_sr_accuracy, masked_sr_loss, \
                        pointer_cd_simi, pointer_cd_loss, masked_sur_accuracy, masked_sur_loss, \
                        shared_nd_accuracy, shared_nd_loss]

    eval_op = [masked_lm_accuracy_op, masked_lm_loss_op, next_sentence_op, next_sentence_loss_op, \
                  adr_recog_accuracy_op, adr_recog_loss_op, masked_sr_accuracy_op, masked_sr_loss_op, \
                  pointer_cd_simi_op, pointer_cd_loss_op, masked_sur_accuracy_op, masked_sur_loss_op, \
                  shared_nd_accuracy_op, shared_nd_loss_op]

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        for epoch in range(FLAGS.num_train_epochs):
            sess.run(iterator.initializer, feed_dict={filenames: [FLAGS.input_file]})
            run_epoch(epoch, sess, saver, FLAGS.output_dir, epoch_save_step, FLAGS.mid_save_step, 
                        input_ids, eval_metrics, total_loss, train_op, eval_op)


if __name__ == "__main__":
    tf.app.run()
