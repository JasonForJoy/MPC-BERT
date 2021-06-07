# coding=utf-8
"""Create (MLM + NSP + RUR + ISS + PCD + MSUR + SND) TF examples for MPC-BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import random
import collections
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import tokenization

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("train_file", './data/ijcai2019/train.json',
                    "Input raw text file.")

flags.DEFINE_string("output_file", './data/pretraining_data.tfrecord',
                    "Output TF example file.")

flags.DEFINE_string("vocab_file", './uncased_L-12_H-768_A-12/vocab.txt',
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_bool("do_lower_case", True,
                  "Whether to lower case the input text. Should be True for uncased "
                  "models and False for cased models.")

flags.DEFINE_integer("max_seq_length", 230, 
                     "Maximum sequence length.")

flags.DEFINE_integer("max_utr_length", 30, 
                     "Maximum single utterance length.")

flags.DEFINE_integer("max_utr_num", 7, 
                     "Maximum utterance number.")

flags.DEFINE_integer("max_predictions_per_seq", 25,
                     "Maximum number of masked LM predictions per sequence.")

flags.DEFINE_integer("max_predictions_per_seq_ar", 4,
                     "Maximum number of Reply-to Utterance Recognition (RUR) predictions per sequence.")

flags.DEFINE_integer("max_predictions_per_seq_sr", 2,
                     "Maximum number of Identical Speaker Searching (ISS) predictions per sequence.")

flags.DEFINE_integer("max_predictions_per_seq_cd", 2,
                     "Maximum number of Pointer Consistency Distinction (PCD) predictions per sequence.")

flags.DEFINE_integer("random_seed", 12345, 
                     "Random seed for data generation.")

flags.DEFINE_integer("dupe_factor", 10,
                     "Number of times to duplicate the input data (with different masks).")

flags.DEFINE_float("masked_lm_prob", 0.15, 
                   "Masked LM probability.")



def print_configuration_op(FLAGS):
  print('My Configurations:')
  for name, value in FLAGS.__flags.items():
    value = value.value
    if type(value) == float:
      print(' %s:\t %f' % (name, value))
    elif type(value) == int:
      print(' %s:\t %d' % (name, value))
    elif type(value) == str:
      print(' %s:\t %s' % (name, value))
    elif type(value) == bool:
      print(' %s:\t %s' % (name, value))
    else:
      print('%s:\t %s' % (name, value))
  print('End of configuration')


class TrainingInstance(object):
  """A single training instance."""

  def __init__(self, tokens_mlm_nsp, segment_ids_mlm_nsp, speaker_ids_mlm_nsp, is_random_next, masked_lm_positions, masked_lm_labels, 
               cls_positions, tokens_ar_msr_pcd, segment_ids_ar_msr_pcd, speaker_ids_ar_msr_pcd, 
               adr_recog_positions, adr_recog_labels, masked_sr_positions, masked_sr_labels, 
               pointer_cd_positions_spk1, pointer_cd_positions_adr1, pointer_cd_positions_spk2, pointer_cd_positions_adr2, pointer_cd_positions_spk3, pointer_cd_positions_adr3, 
               tokens_msur, segment_ids_msur, speaker_ids_msur, masked_sur_positions, masked_sur_labels,
               tokens_snd, segment_ids_snd, speaker_ids_snd, is_random_next_snd):

    self.tokens_mlm_nsp = tokens_mlm_nsp
    self.segment_ids_mlm_nsp = segment_ids_mlm_nsp
    self.speaker_ids_mlm_nsp = speaker_ids_mlm_nsp
    self.is_random_next = is_random_next
    self.masked_lm_positions = masked_lm_positions
    self.masked_lm_labels = masked_lm_labels

    self.cls_positions = cls_positions
    self.tokens_ar_msr_pcd = tokens_ar_msr_pcd
    self.segment_ids_ar_msr_pcd = segment_ids_ar_msr_pcd
    self.speaker_ids_ar_msr_pcd = speaker_ids_ar_msr_pcd

    self.adr_recog_positions = adr_recog_positions
    self.adr_recog_labels = adr_recog_labels

    self.masked_sr_positions = masked_sr_positions
    self.masked_sr_labels = masked_sr_labels

    self.pointer_cd_positions_spk1 = pointer_cd_positions_spk1
    self.pointer_cd_positions_adr1 = pointer_cd_positions_adr1
    self.pointer_cd_positions_spk2 = pointer_cd_positions_spk2
    self.pointer_cd_positions_adr2 = pointer_cd_positions_adr2
    self.pointer_cd_positions_spk3 = pointer_cd_positions_spk3
    self.pointer_cd_positions_adr3 = pointer_cd_positions_adr3

    self.tokens_msur = tokens_msur
    self.segment_ids_msur = segment_ids_msur
    self.speaker_ids_msur = speaker_ids_msur
    self.masked_sur_positions = masked_sur_positions
    self.masked_sur_labels = masked_sur_labels

    self.tokens_snd = tokens_snd
    self.segment_ids_snd = segment_ids_snd
    self.speaker_ids_snd = speaker_ids_snd
    self.is_random_next_snd = is_random_next_snd

  def __str__(self):
    s = ""
    s += "tokens_mlm_nsp: %s\n" % (" ".join([tokenization.printable_text(x) for x in self.tokens_mlm_nsp]))
    s += "segment_ids_mlm_nsp: %s\n" % (" ".join([str(x) for x in self.segment_ids_mlm_nsp]))
    s += "speaker_ids_mlm_nsp: %s\n" % (" ".join([str(x) for x in self.speaker_ids_mlm_nsp]))
    s += "is_random_next: %s\n" % self.is_random_next 
    s += "masked_lm_positions: %s\n" % (" ".join([str(x) for x in self.masked_lm_positions]))
    s += "masked_lm_labels: %s\n" % (" ".join([tokenization.printable_text(x) for x in self.masked_lm_labels]))

    s += "cls_positions: %s\n" % (" ".join([str(x) for x in self.cls_positions]))
    s += "tokens_ar_msr_pcd: %s\n" % (" ".join([tokenization.printable_text(x) for x in self.tokens_ar_msr_pcd]))
    s += "segment_ids_ar_msr_pcd: %s\n" % (" ".join([str(x) for x in self.segment_ids_ar_msr_pcd]))
    s += "speaker_ids_ar_msr_pcd: %s\n" % (" ".join([str(x) for x in self.speaker_ids_ar_msr_pcd]))

    s += "adr_recog_positions: %s\n" % (" ".join([str(x) for x in self.adr_recog_positions]))
    s += "adr_recog_labels: %s\n" % (" ".join([str(x) for x in self.adr_recog_labels]))

    s += "masked_sr_positions: %s\n" % (" ".join([str(x) for x in self.masked_sr_positions]))
    s += "masked_sr_labels: %s\n" % (" ".join([str(x) for x in self.masked_sr_labels]))

    s += "pointer_cd_positions_spk1: %s\n" % (" ".join([str(x) for x in self.pointer_cd_positions_spk1]))
    s += "pointer_cd_positions_adr1: %s\n" % (" ".join([str(x) for x in self.pointer_cd_positions_adr1]))
    s += "pointer_cd_positions_spk2: %s\n" % (" ".join([str(x) for x in self.pointer_cd_positions_spk2]))
    s += "pointer_cd_positions_adr2: %s\n" % (" ".join([str(x) for x in self.pointer_cd_positions_adr2]))
    s += "pointer_cd_positions_spk3: %s\n" % (" ".join([str(x) for x in self.pointer_cd_positions_spk3]))
    s += "pointer_cd_positions_adr3: %s\n" % (" ".join([str(x) for x in self.pointer_cd_positions_adr3]))

    s += "tokens_msur: %s\n" % (" ".join([tokenization.printable_text(x) for x in self.tokens_msur]))
    s += "segment_ids_msur: %s\n" % (" ".join([str(x) for x in self.segment_ids_msur]))
    s += "speaker_ids_msur: %s\n" % (" ".join([str(x) for x in self.speaker_ids_msur]))
    s += "masked_sur_positions: %s\n" % (" ".join([str(x) for x in self.masked_sur_positions]))
    s += "masked_sur_labels: %s\n" % (" ".join([tokenization.printable_text(x) for x in self.masked_sur_labels]))

    s += "tokens_snd: %s\n" % (" ".join([tokenization.printable_text(x) for x in self.tokens_snd]))
    s += "segment_ids_snd: %s\n" % (" ".join([str(x) for x in self.segment_ids_snd]))
    s += "speaker_ids_snd: %s\n" % (" ".join([str(x) for x in self.speaker_ids_snd]))
    s += "is_random_next_snd: %s\n" % self.is_random_next_snd
    s += "\n"
    return s

  def __repr__(self):
    return self.__str__()


def get_threads_pair_list(ctx_list, ctx_spk_list, ctx_relation_list, rsp_list, rsp_spk_list, rsp_relation_list):

  tf.logging.info("Extracting matched thread pairs ...")

  def get_tgts(src, src_to_tgts):
    if src in src_to_tgts:
      return src_to_tgts[src]
    else:
      return []

  thread_a_list = []
  thread_a_spk_list = []
  thread_b_list = []
  thread_b_spk_list = []
  for i in tqdm(range(len(ctx_list))):

    ctx = ctx_list[i]
    ctx_spk = ctx_spk_list[i]
    ctx_relation = ctx_relation_list[i]

    rsp = rsp_list[i]
    rsp_spk = rsp_spk_list[i]
    rsp_relation = rsp_relation_list[i]

    integrate_ctx_relation = ctx_relation + [[len(ctx), rsp_relation]]
    integrate_ctx = ctx + [rsp]
    integrate_ctx_spk = ctx_spk + [rsp_spk]
    assert len(integrate_ctx) == len(integrate_ctx_spk)

    src_to_tgts = collections.OrderedDict()
    for relation in integrate_ctx_relation:
      tgt, src = relation
      if src not in src_to_tgts:
        src_to_tgts[src] = [tgt]
      else:
        src_to_tgts[src].append(tgt)

    threads = []
    for src, tgts in src_to_tgts.items():
      if len(tgts) > 1:
        # init
        src_to_seq = {}
        for tgt in tgts:
          src_to_seq[tgt] = [tgt]

        # recurrent
        for new_src, seq in src_to_seq.items():
          new_tgts = get_tgts(new_src, src_to_tgts)
          seq.extend(new_tgts)
          while len(new_tgts) > 0:
            follow = []
            for new_tgt in new_tgts:
              follow.extend(get_tgts(new_tgt, src_to_tgts))
            seq.extend(follow)
            new_tgts = follow

        for _, seq in src_to_seq.items():
          threads.append(sorted(seq))
        break  # break to select only the first shared node

    # select top-2 length thread
    threads_len = [len(thread) for thread in threads]
    thread_a = threads[np.argmax(np.array(threads_len))]
    threads.remove(thread_a)
    threads_len = [len(thread) for thread in threads]
    thread_b = threads[np.argmax(np.array(threads_len))]
    threads_pair = sorted([thread_a, thread_b])

    thread_a = [integrate_ctx[utr_id] for utr_id in threads_pair[0]]
    thread_a_spk = [integrate_ctx_spk[utr_id] for utr_id in threads_pair[0]]
    assert len(thread_a) == len(thread_a_spk)

    thread_b = [integrate_ctx[utr_id] for utr_id in threads_pair[1]]
    thread_b_spk = [integrate_ctx_spk[utr_id] for utr_id in threads_pair[1]]
    assert len(thread_b) == len(thread_b_spk)

    thread_a_list.append(thread_a)
    thread_a_spk_list.append(thread_a_spk)
    thread_b_list.append(thread_b)
    thread_b_spk_list.append(thread_b_spk)

  return thread_a_list, thread_a_spk_list, thread_b_list, thread_b_spk_list


def create_training_instances(ctx_list, ctx_spk_list, ctx_adr_list, ctx_relation_list, rsp_list, rsp_spk_list, rsp_adr_list, rsp_relation_list, 
                              tokenizer, max_seq_length, max_utr_length, dupe_factor, masked_lm_prob, 
                              max_predictions_per_seq, max_predictions_per_seq_ar, max_predictions_per_seq_sr, max_predictions_per_seq_cd, rng):

  vocab_words = list(tokenizer.vocab.keys())

  sid_r = np.arange(0, len(ctx_list))
  rng.shuffle(sid_r)

  thread_a_list, thread_a_spk_list, thread_b_list, thread_b_spk_list = get_threads_pair_list(
      ctx_list, ctx_spk_list, ctx_relation_list, rsp_list, rsp_spk_list, rsp_relation_list)

  instances = []
  for _ in tqdm(range(dupe_factor)):
    for i in tqdm(range(len(ctx_list))):

      # context
      ctx = ctx_list[i]
      ctx_spk = ctx_spk_list[i]
      ctx_adr = ctx_adr_list[i]
      ctx_relation = ctx_relation_list[i]
      ctx_tokens = []
      for utr in ctx:
        utr = tokenization.convert_to_unicode(utr)
        utr_tokens = tokenizer.tokenize(utr)
        ctx_tokens.append(utr_tokens)
      assert len(ctx_tokens) == len(ctx_spk)

      # positive and negative response
      rsp_pos = rsp_list[i]
      rsp_neg = rsp_list[sid_r[i]]
      rsp_spk = rsp_spk_list[i]
      rsp_adr = rsp_adr_list[i]
      rsp_relation = rsp_relation_list[i]
      rsp_pos = tokenization.convert_to_unicode(rsp_pos)
      rsp_pos_tokens = tokenizer.tokenize(rsp_pos)
      rsp_neg = tokenization.convert_to_unicode(rsp_neg)
      rsp_neg_tokens = tokenizer.tokenize(rsp_neg)

      # match/mismatched thread pair
      thread_a = thread_a_list[i]
      thread_a_spk = thread_a_spk_list[i]
      thread_a_tokens = []
      for utr in thread_a:
        utr = tokenization.convert_to_unicode(utr)
        utr_tokens = tokenizer.tokenize(utr)
        thread_a_tokens.append(utr_tokens)
      assert len(thread_a_tokens) == len(thread_a_spk)

      if random.random() < 0.5:
        thread_b = thread_b_list[sid_r[i]]
        thread_b_spk = thread_b_spk_list[sid_r[i]]
        is_random_next_snd = True
      else:
        thread_b = thread_b_list[i]
        thread_b_spk = thread_b_spk_list[i]
        is_random_next_snd = False
      thread_b_tokens = []
      for utr in thread_b:
        utr = tokenization.convert_to_unicode(utr)
        utr_tokens = tokenizer.tokenize(utr)
        thread_b_tokens.append(utr_tokens)
      assert len(thread_b_tokens) == len(thread_b_spk)

      instance = create_instances_from_document(
          ctx_tokens, ctx_spk, ctx_adr, ctx_relation, rsp_pos_tokens, rsp_neg_tokens, rsp_spk, rsp_adr, rsp_relation, 
          thread_a_tokens, thread_a_spk, thread_b_tokens, thread_b_spk, is_random_next_snd, 
          max_seq_length, max_utr_length, masked_lm_prob, 
          max_predictions_per_seq, max_predictions_per_seq_ar, max_predictions_per_seq_sr, max_predictions_per_seq_cd, vocab_words, rng)
      instances.append(instance)
      if i < 5:
        print(instance)

  rng.shuffle(instances)
  return instances


def create_instances_from_document(
    ctx_tokens, ctx_spk, ctx_adr, ctx_relation, rsp_pos_tokens, rsp_neg_tokens, rsp_spk, rsp_adr, rsp_relation, 
    thread_a_tokens, thread_a_spk, thread_b_tokens, thread_b_spk, is_random_next_snd, 
    max_seq_length, max_utr_length, masked_lm_prob, 
    max_predictions_per_seq, max_predictions_per_seq_ar, max_predictions_per_seq_sr, max_predictions_per_seq_cd, vocab_words, rng):
  """Creates `TrainingInstance`s for a single document."""

  # Account for [CLS]s, [CLS]/[SEP], [SEP]
  max_num_tokens = max_seq_length - len(ctx_tokens) - 2
  truncate_seq_pair(ctx_tokens, rsp_pos_tokens, max_num_tokens, rng)
  truncate_seq_pair(ctx_tokens, rsp_neg_tokens, max_num_tokens, rng)
  for utr_tokens in ctx_tokens:
    assert len(utr_tokens) >= 1
  assert len(rsp_pos_tokens) >= 1
  assert len(rsp_neg_tokens) >= 1

  tokens = []
  segment_ids = []
  speaker_ids = []
  cls_positions = []
  
  # context
  for i in range(len(ctx_tokens)):

    utr_tokens = ctx_tokens[i]
    utr_spk = ctx_spk[i]

    cls_positions.append(len(tokens))
    tokens.append("[CLS]")
    segment_ids.append(0)
    speaker_ids.append(utr_spk)

    for token in utr_tokens:
      tokens.append(token)
      segment_ids.append(0)
      speaker_ids.append(utr_spk)

  # MLM and NSP share the same input
  (tokens_mlm_nsp, segment_ids_mlm_nsp, speaker_ids_mlm_nsp, is_random_next) = create_next_sentence_predictions(
      tokens, segment_ids, speaker_ids, rsp_pos_tokens, rsp_neg_tokens, rsp_spk)

  (tokens_mlm_nsp, masked_lm_positions, masked_lm_labels) = create_masked_lm_predictions(
      tokens_mlm_nsp, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)


  # response
  cls_positions.append(len(tokens))
  tokens.append("[CLS]")
  segment_ids.append(0)
  speaker_ids.append(rsp_spk)

  for token in rsp_pos_tokens:
    tokens.append(token)
    segment_ids.append(0)
    speaker_ids.append(rsp_spk)

  tokens.append("[SEP]")
  segment_ids.append(0)
  speaker_ids.append(rsp_spk)

  # integrate respone into context
  if rsp_relation > -1:
    integrate_ctx_relation = ctx_relation + [[len(ctx_tokens), rsp_relation]]
  if rsp_spk > -1:
    integrate_ctx_spk = ctx_spk + [rsp_spk]
  if rsp_adr > -1:
    integrate_ctx_adr = ctx_adr + [rsp_adr]


  # RUR, ISS and PCD share the same input
  tokens_ar_msr_pcd = list(tokens)
  segment_ids_ar_msr_pcd = list(segment_ids)
  speaker_ids_ar_msr_pcd = list(speaker_ids)

  (adr_recog_positions, adr_recog_labels) = create_replyto_utterance_recognition_predictions(
      integrate_ctx_relation, max_predictions_per_seq_ar, rng)

  (speaker_ids_ar_msr_pcd, masked_sr_positions, masked_sr_labels) = create_identical_speaker_searching_predictions(
      speaker_ids_ar_msr_pcd, integrate_ctx_spk, cls_positions, max_predictions_per_seq_sr, rng)

  (pointer_cd_positions_spk1, pointer_cd_positions_adr1, pointer_cd_positions_spk2, pointer_cd_positions_adr2, \
      pointer_cd_positions_spk3, pointer_cd_positions_adr3) = create_pointer_consistency_distinction_predictions(
          integrate_ctx_spk, integrate_ctx_adr, integrate_ctx_relation, max_predictions_per_seq_cd, rng)
      

  tokens_msur = list(tokens)
  segment_ids_msur = list(segment_ids)
  speaker_ids_msur = list(speaker_ids)
  (tokens_msur, masked_sur_positions, masked_sur_labels) = create_masked_shared_utterance_restoration_predictions(
      tokens_msur, integrate_ctx_relation, cls_positions, max_utr_length, vocab_words, rng)

  tokens_snd, segment_ids_snd, speaker_ids_snd = create_shared_node_detection_predictions(
      thread_a_tokens, thread_a_spk, thread_b_tokens, thread_b_spk, max_seq_length, rng)

  instance = TrainingInstance(
        tokens_mlm_nsp=tokens_mlm_nsp,
        segment_ids_mlm_nsp=segment_ids_mlm_nsp,
        speaker_ids_mlm_nsp=speaker_ids_mlm_nsp,
        is_random_next=is_random_next,
        masked_lm_positions=masked_lm_positions,
        masked_lm_labels=masked_lm_labels,

        cls_positions=cls_positions,
        tokens_ar_msr_pcd=tokens_ar_msr_pcd,
        segment_ids_ar_msr_pcd=segment_ids_ar_msr_pcd,
        speaker_ids_ar_msr_pcd=speaker_ids_ar_msr_pcd,

        adr_recog_positions=adr_recog_positions,
        adr_recog_labels=adr_recog_labels,

        masked_sr_positions=masked_sr_positions,
        masked_sr_labels=masked_sr_labels,

        pointer_cd_positions_spk1=pointer_cd_positions_spk1,
        pointer_cd_positions_adr1=pointer_cd_positions_adr1,
        pointer_cd_positions_spk2=pointer_cd_positions_spk2,
        pointer_cd_positions_adr2=pointer_cd_positions_adr2,
        pointer_cd_positions_spk3=pointer_cd_positions_spk3,
        pointer_cd_positions_adr3=pointer_cd_positions_adr3,

        tokens_msur=tokens_msur,
        segment_ids_msur=segment_ids_msur,
        speaker_ids_msur=speaker_ids_msur,
        masked_sur_positions=masked_sur_positions,
        masked_sur_labels=masked_sur_labels,

        tokens_snd=tokens_snd,
        segment_ids_snd=segment_ids_snd,
        speaker_ids_snd=speaker_ids_snd,
        is_random_next_snd=is_random_next_snd,
        )

  return instance


def create_next_sentence_predictions(tokens, segment_ids, speaker_ids, rsp_pos_tokens, rsp_neg_tokens, rsp_spk):
  """Creates the predictions for the Next Sentence Prediction (NSP) objective."""

  tokens_nsp = list(tokens)
  segment_ids_nsp = list(segment_ids)
  speaker_ids_nsp = list(speaker_ids)

  if random.random() < 0.5:
    rsp_tokens = rsp_neg_tokens
    is_random_next = True
  else:
    rsp_tokens = rsp_pos_tokens
    is_random_next = False

  tokens_nsp.append("[SEP]")
  segment_ids_nsp.append(0)
  speaker_ids_nsp.append(0)

  for token in rsp_tokens:
    tokens_nsp.append(token)
    segment_ids_nsp.append(1)
    speaker_ids_nsp.append(rsp_spk)

  tokens_nsp.append("[SEP]")
  segment_ids_nsp.append(1)
  speaker_ids_nsp.append(rsp_spk)

  return tokens_nsp, segment_ids_nsp, speaker_ids_nsp, is_random_next


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])

def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng):
  """Creates the predictions for the Masked Language Model (MLM) objective."""

  cand_indexes = []
  for (i, token) in enumerate(tokens):
    if token == "[CLS]" or token == "[SEP]":
      continue
    cand_indexes.append(i)
  rng.shuffle(cand_indexes)

  output_tokens = list(tokens)

  num_to_predict = min(max_predictions_per_seq,
                       max(1, int(round(len(tokens) * masked_lm_prob))))

  masked_lms = []
  covered_indexes = set()
  for index in cand_indexes:
    if len(masked_lms) >= num_to_predict:
      break
    if index in covered_indexes:
      continue
    covered_indexes.add(index)

    masked_token = None
    # 80% of the time, replace with [MASK]
    if rng.random() < 0.8:
      masked_token = "[MASK]"
    else:
      # 10% of the time, keep original
      if rng.random() < 0.5:
        masked_token = tokens[index]
      # 10% of the time, replace with random word
      else:
        masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

    output_tokens[index] = masked_token

    masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

  masked_lms = sorted(masked_lms, key=lambda x: x.index)

  masked_lm_positions = []
  masked_lm_labels = []
  for p in masked_lms:
    masked_lm_positions.append(p.index)
    masked_lm_labels.append(p.label)

  return (output_tokens, masked_lm_positions, masked_lm_labels)


def create_replyto_utterance_recognition_predictions(ctx_relation, max_predictions_per_seq_ar, rng):
  """Creates the predictions for the Reply-to Utterance Recognition (RUR) objective."""
  
  cand_relations = list(ctx_relation)
  rng.shuffle(cand_relations)

  adr_recog_positions = []
  adr_recog_labels = []
  for relation in cand_relations:
    if len(adr_recog_positions) >= max_predictions_per_seq_ar:
      break
    tgt, src = relation
    adr_recog_positions.append(tgt)
    adr_recog_labels.append(src)

  return (adr_recog_positions, adr_recog_labels)


def create_identical_speaker_searching_predictions(speaker_ids, ctx_spk, cls_positions, max_predictions_per_seq_sr, rng):
  """Creates the predictions for the Identical Speaker Searching (ISS) objective."""

  utrs_with_same_spk = {}
  for utr, spk in enumerate(ctx_spk):
    if spk in utrs_with_same_spk:
      utrs_with_same_spk[spk].append(utr)
    else:
      utrs_with_same_spk[spk] = [utr]

  cand_utrs_with_same_spk = []
  for spk, utrs in utrs_with_same_spk.items():
    if len(utrs) < 2:
      continue
    cand_utrs_with_same_spk.append(utrs)
  rng.shuffle(cand_utrs_with_same_spk)

  masked_sr_positions = []
  masked_sr_labels = []
  for utrs in cand_utrs_with_same_spk:
    if len(masked_sr_positions) >= max_predictions_per_seq_sr:
      break

    label_masked = random.sample(utrs, 2)
    label_masked = sorted(label_masked)
    label_utr = label_masked[0]
    masked_utr = label_masked[1]

    start = cls_positions[masked_utr]
    end = cls_positions[masked_utr + 1] if masked_utr + 1 < len(cls_positions) else len(speaker_ids)
    for index in range(start, end):
      speaker_ids[index] = 0

    masked_sr_positions.append(masked_utr)
    masked_sr_labels.append(label_utr)

  return speaker_ids, masked_sr_positions, masked_sr_labels


def create_pointer_consistency_distinction_predictions(ctx_spk, ctx_adr, ctx_relation, max_predictions_per_seq_cd, rng):
  """Creates the predictions for the Pointer Consistency Distinction (PCD) objective."""

  spk_adr_to_utrs = {}
  for utr, spk_adr in enumerate(zip(ctx_spk, ctx_adr)):
    if spk_adr in spk_adr_to_utrs:
      spk_adr_to_utrs[spk_adr].append(utr)
    else:
      spk_adr_to_utrs[spk_adr] = [utr]

  cand_spk_adr_to_utrs = []
  for spk_adr, utrs in spk_adr_to_utrs.items():
    if len(utrs) < 2:
      continue
    cand_spk_adr_to_utrs.append((spk_adr, utrs))
  rng.shuffle(cand_spk_adr_to_utrs)

  spk_to_utrs = {}
  for utr, spk in enumerate(ctx_spk):
    if spk in spk_to_utrs:
      spk_to_utrs[spk].append(utr)
    else:
      spk_to_utrs[spk] = [utr]

  pointer_cd_positions_spk1 = []
  pointer_cd_positions_adr1 = []
  pointer_cd_positions_spk2 = []
  pointer_cd_positions_adr2 = []
  pointer_cd_positions_spk3 = []
  pointer_cd_positions_adr3 = []
  for spk_adr, utrs in cand_spk_adr_to_utrs:
    if len(pointer_cd_positions_spk1) >= max_predictions_per_seq_cd:
      break
    (spk, adr) = spk_adr

    # positive sample
    positive_relation = []
    for spk_utr in utrs:
      adr_utr = -1
      for adr_utr_tmp in spk_to_utrs[adr]:
        if adr_utr_tmp < spk_utr and adr_utr_tmp > adr_utr:
          adr_utr = adr_utr_tmp
      positive_relation.append([spk_utr, adr_utr])

    positive_relation_sampled = random.sample(positive_relation, 2)
    [spk_utr1, adr_utr1] = positive_relation_sampled[0]
    pointer_cd_positions_spk1.append(spk_utr1)
    pointer_cd_positions_adr1.append(adr_utr1)
    [spk_utr2, adr_utr2] = positive_relation_sampled[1]
    pointer_cd_positions_spk2.append(spk_utr2)
    pointer_cd_positions_adr2.append(adr_utr2)

    # negative sample
    cand_ctx_relation = []
    for relation in ctx_relation:
      if relation in positive_relation:
        continue
      cand_ctx_relation.append(relation)

    if len(cand_ctx_relation) == 0:
      return [], [], [], [], [], [], 

    assert len(cand_ctx_relation) > 0
    rng.shuffle(cand_ctx_relation)
    neg_pointer = cand_ctx_relation[0]
    spk_utr3, adr_utr3 = neg_pointer[0], neg_pointer[1]
    pointer_cd_positions_spk3.append(spk_utr3)
    pointer_cd_positions_adr3.append(adr_utr3)

  return pointer_cd_positions_spk1, pointer_cd_positions_adr1, pointer_cd_positions_spk2, pointer_cd_positions_adr2, \
         pointer_cd_positions_spk3, pointer_cd_positions_adr3


def create_masked_shared_utterance_restoration_predictions(tokens, ctx_relation, cls_positions, max_utr_length, vocab_words, rng):
  """Creates the predictions for the Masked Shared Utterance Restoration (MSUR) objective."""

  src_to_num_tgt = {}
  for relation in ctx_relation:
    tgt, src = relation
    if src not in src_to_num_tgt:
      src_to_num_tgt[src] = 1
    else:
      src_to_num_tgt[src] += 1

  shared_src = []
  for src, num_tgt in src_to_num_tgt.items():
    if num_tgt > 1:
      shared_src.append(src)

  if len(shared_src) == 0:
    return (tokens, [], [])

  rng.shuffle(shared_src)
  masked_utr_id = shared_src[0]
  masked_start = cls_positions[masked_utr_id]
  if (cls_positions[masked_utr_id + 1] - cls_positions[masked_utr_id]) <= max_utr_length:
    masked_end = cls_positions[masked_utr_id + 1] 
  else:
    masked_end = cls_positions[masked_utr_id] + max_utr_length

  output_tokens = list(tokens)
  masked_token = "[MASK]"
  masked_sur_positions = []
  masked_sur_labels = []
  for index in range(masked_start, masked_end):
    output_tokens[index] = masked_token
    masked_sur_positions.append(index)
    masked_sur_labels.append(tokens[index])

  return (output_tokens, masked_sur_positions, masked_sur_labels)


def create_shared_node_detection_predictions(thread_a_tokens, thread_a_spk, thread_b_tokens, thread_b_spk, max_seq_length, rng):
  """Creates the predictions for the Shared Node Detection (SND) objective."""

  # Account for [CLS], [SEP], [SEP]
  max_num_tokens = max_seq_length - 3
  truncate_seq_list_pair(thread_a_tokens, thread_b_tokens, max_num_tokens, rng)
  for utr_tokens in thread_a_tokens:
    assert len(utr_tokens) >= 1
  for utr_tokens in thread_b_tokens:
    assert len(utr_tokens) >= 1

  tokens = []
  segment_ids = []
  speaker_ids = []
  
  tokens.append("[CLS]")
  segment_ids.append(0)
  speaker_ids.append(0)

  # thread_a
  for i in range(len(thread_a_tokens)):
    utr_tokens = thread_a_tokens[i]
    utr_spk = thread_a_spk[i]

    for token in utr_tokens:
      tokens.append(token)
      segment_ids.append(0)
      speaker_ids.append(utr_spk)

  tokens.append("[SEP]")
  segment_ids.append(0)
  speaker_ids.append(utr_spk)

  # thread_b
  for i in range(len(thread_b_tokens)):
    utr_tokens = thread_b_tokens[i]
    utr_spk = thread_b_spk[i]

    for token in utr_tokens:
      tokens.append(token)
      segment_ids.append(1)
      speaker_ids.append(utr_spk)

  tokens.append("[SEP]")
  segment_ids.append(1)
  speaker_ids.append(utr_spk)

  return tokens, segment_ids, speaker_ids
  

def truncate_seq_pair(ctx_tokens, rsp_tokens, max_num_tokens, rng):
  """Truncates a single sequence to a maximum sequence length."""
  while True:
    utr_lens = [len(utr_tokens) for utr_tokens in ctx_tokens]
    total_length = sum(utr_lens) + len(rsp_tokens)
    if total_length <= max_num_tokens:
      break

    # truncate the longest utterance or response
    if sum(utr_lens) > len(rsp_tokens):
      trunc_tokens = ctx_tokens[np.argmax(np.array(utr_lens))]
    else:
      trunc_tokens = rsp_tokens
    assert len(trunc_tokens) >= 1

    # We want to sometimes truncate from the front and sometimes from the
    # back to add more randomness and avoid biases.
    if rng.random() < 0.5:
      del trunc_tokens[0]
    else:
      trunc_tokens.pop()


def truncate_seq_list_pair(A_tokens, B_tokens, max_num_tokens, rng):
  """Truncates a single sequence to a maximum sequence length."""
  while True:
    A_lens = [len(a_tokens) for a_tokens in A_tokens]
    B_lens = [len(b_tokens) for b_tokens in B_tokens]
    total_length = sum(A_lens) + sum(B_lens)
    if total_length <= max_num_tokens:
      break

    # truncate the longest one
    if sum(A_lens) > sum(B_lens):
      trunc_tokens = A_tokens[np.argmax(np.array(A_lens))]
    else:
      trunc_tokens = B_tokens[np.argmax(np.array(B_lens))]
    assert len(trunc_tokens) >= 1

    # We want to sometimes truncate from the front and sometimes from the
    # back to add more randomness and avoid biases.
    if rng.random() < 0.5:
      del trunc_tokens[0]
    else:
      trunc_tokens.pop()


def write_instance_to_example_files(instances, tokenizer, max_seq_length, max_utr_length, max_utr_num, 
                                    max_predictions_per_seq, max_predictions_per_seq_ar, max_predictions_per_seq_sr, 
                                    max_predictions_per_seq_cd, output_files):
  """Create TF example files from `TrainingInstance`s."""
  writers = []
  for output_file in output_files:
    writers.append(tf.python_io.TFRecordWriter(output_file))

  writer_index = 0

  total_written = 0
  for (inst_index, instance) in enumerate(instances):

    input_ids_mlm_nsp = tokenizer.convert_tokens_to_ids(instance.tokens_mlm_nsp)
    input_mask_mlm_nsp = [1] * len(input_ids_mlm_nsp)
    segment_ids_mlm_nsp = list(instance.segment_ids_mlm_nsp)
    speaker_ids_mlm_nsp = list(instance.speaker_ids_mlm_nsp)
    assert len(input_ids_mlm_nsp) <= max_seq_length
    while len(input_ids_mlm_nsp) < max_seq_length:
      input_ids_mlm_nsp.append(0)
      input_mask_mlm_nsp.append(0)
      segment_ids_mlm_nsp.append(0)
      speaker_ids_mlm_nsp.append(0)
    assert len(input_ids_mlm_nsp) == max_seq_length
    assert len(input_mask_mlm_nsp) == max_seq_length
    assert len(segment_ids_mlm_nsp) == max_seq_length
    assert len(speaker_ids_mlm_nsp) == max_seq_length

    # NSP
    next_sentence_label = 1 if instance.is_random_next else 0

    # MLM
    masked_lm_positions = list(instance.masked_lm_positions)
    masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
    masked_lm_weights = [1.0] * len(masked_lm_ids)
    assert len(masked_lm_positions) <= max_predictions_per_seq
    while len(masked_lm_positions) < max_predictions_per_seq:
      masked_lm_positions.append(0)
      masked_lm_ids.append(0)
      masked_lm_weights.append(0.0)


    cls_positions = list(instance.cls_positions)
    assert len(cls_positions) <= max_utr_num
    while len(cls_positions) < max_utr_num:
      cls_positions.append(0)

    input_ids_ar_msr_pcd = tokenizer.convert_tokens_to_ids(instance.tokens_ar_msr_pcd)
    input_mask_ar_msr_pcd = [1] * len(input_ids_ar_msr_pcd)
    segment_ids_ar_msr_pcd = list(instance.segment_ids_ar_msr_pcd)
    speaker_ids_ar_msr_pcd = list(instance.speaker_ids_ar_msr_pcd)
    assert len(input_ids_ar_msr_pcd) <= max_seq_length
    while len(input_ids_ar_msr_pcd) < max_seq_length:
      input_ids_ar_msr_pcd.append(0)
      input_mask_ar_msr_pcd.append(0)
      segment_ids_ar_msr_pcd.append(0)
      speaker_ids_ar_msr_pcd.append(0)
    assert len(input_ids_ar_msr_pcd) == max_seq_length
    assert len(input_mask_ar_msr_pcd) == max_seq_length
    assert len(segment_ids_ar_msr_pcd) == max_seq_length
    assert len(speaker_ids_ar_msr_pcd) == max_seq_length

    # RUR
    adr_recog_positions = list(instance.adr_recog_positions)
    adr_recog_labels = list(instance.adr_recog_labels)
    adr_recog_weights = [1.0] * len(adr_recog_labels)
    assert len(adr_recog_positions) <= max_predictions_per_seq_ar
    while len(adr_recog_positions) < max_predictions_per_seq_ar:
      adr_recog_positions.append(0)
      adr_recog_labels.append(0)
      adr_recog_weights.append(0.0)

    # ISS
    masked_sr_positions = list(instance.masked_sr_positions)
    masked_sr_labels = list(instance.masked_sr_labels)
    masked_sr_weights = [1.0] * len(masked_sr_labels)
    assert len(masked_sr_positions) <= max_predictions_per_seq_sr
    while len(masked_sr_positions) < max_predictions_per_seq_sr:
      masked_sr_positions.append(0)
      masked_sr_labels.append(0)
      masked_sr_weights.append(0.0)

    # PCD
    pointer_cd_positions_spk1 = list(instance.pointer_cd_positions_spk1)
    pointer_cd_positions_adr1 = list(instance.pointer_cd_positions_adr1)
    pointer_cd_positions_spk2 = list(instance.pointer_cd_positions_spk2)
    pointer_cd_positions_adr2 = list(instance.pointer_cd_positions_adr2)
    pointer_cd_positions_spk3 = list(instance.pointer_cd_positions_spk3)
    pointer_cd_positions_adr3 = list(instance.pointer_cd_positions_adr3)
    pointer_cd_weights = [1.0] * len(pointer_cd_positions_spk1)
    assert len(pointer_cd_positions_spk1) <= max_predictions_per_seq_cd
    while len(pointer_cd_positions_spk1) < max_predictions_per_seq_cd:
      pointer_cd_positions_spk1.append(0)
      pointer_cd_positions_adr1.append(0)
      pointer_cd_positions_spk2.append(0)
      pointer_cd_positions_adr2.append(0)
      pointer_cd_positions_spk3.append(0)
      pointer_cd_positions_adr3.append(0)
      pointer_cd_weights.append(0.0)

    # MSUR
    input_ids_msur = tokenizer.convert_tokens_to_ids(instance.tokens_msur)
    input_mask_msur = [1] * len(input_ids_msur)
    segment_ids_msur = list(instance.segment_ids_msur)
    speaker_ids_msur = list(instance.speaker_ids_msur)
    assert len(input_ids_msur) <= max_seq_length
    while len(input_ids_msur) < max_seq_length:
      input_ids_msur.append(0)
      input_mask_msur.append(0)
      segment_ids_msur.append(0)
      speaker_ids_msur.append(0)
    assert len(input_ids_msur) == max_seq_length
    assert len(input_mask_msur) == max_seq_length
    assert len(segment_ids_msur) == max_seq_length
    assert len(speaker_ids_msur) == max_seq_length

    masked_sur_positions = list(instance.masked_sur_positions)
    masked_sur_ids = tokenizer.convert_tokens_to_ids(instance.masked_sur_labels)
    masked_sur_weights = [1.0] * len(masked_sur_ids)
    assert len(masked_sur_positions) <= max_utr_length
    while len(masked_sur_positions) < max_utr_length:
      masked_sur_positions.append(0)
      masked_sur_ids.append(0)
      masked_sur_weights.append(0.0)

    # SND
    input_ids_snd = tokenizer.convert_tokens_to_ids(instance.tokens_snd)
    input_mask_snd = [1] * len(input_ids_snd)
    segment_ids_snd = list(instance.segment_ids_snd)
    speaker_ids_snd = list(instance.speaker_ids_snd)
    assert len(input_ids_snd) <= max_seq_length
    while len(input_ids_snd) < max_seq_length:
      input_ids_snd.append(0)
      input_mask_snd.append(0)
      segment_ids_snd.append(0)
      speaker_ids_snd.append(0)
    assert len(input_ids_snd) == max_seq_length
    assert len(input_mask_snd) == max_seq_length
    assert len(segment_ids_snd) == max_seq_length
    assert len(speaker_ids_snd) == max_seq_length
    next_thread_label = 1 if instance.is_random_next_snd else 0


    features = collections.OrderedDict()
    features["input_ids_mlm_nsp"] = create_int_feature(input_ids_mlm_nsp)
    features["input_mask_mlm_nsp"] = create_int_feature(input_mask_mlm_nsp)
    features["segment_ids_mlm_nsp"] = create_int_feature(segment_ids_mlm_nsp)
    features["speaker_ids_mlm_nsp"] = create_int_feature(speaker_ids_mlm_nsp)
    features["next_sentence_labels"] = create_int_feature([next_sentence_label])
    features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
    features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
    features["masked_lm_weights"] = create_float_feature(masked_lm_weights)

    features["cls_positions"] = create_int_feature(cls_positions)               # [max_utr_num, ]
    features["input_ids_ar_msr_pcd"] = create_int_feature(input_ids_ar_msr_pcd)
    features["input_mask_ar_msr_pcd"] = create_int_feature(input_mask_ar_msr_pcd)
    features["segment_ids_ar_msr_pcd"] = create_int_feature(segment_ids_ar_msr_pcd)
    features["speaker_ids_ar_msr_pcd"] = create_int_feature(speaker_ids_ar_msr_pcd)

    features["adr_recog_positions"] = create_int_feature(adr_recog_positions)   # [max_predictions_per_seq_ar, ]
    features["adr_recog_labels"] = create_int_feature(adr_recog_labels)         # [max_predictions_per_seq_ar, ]
    features["adr_recog_weights"] = create_float_feature(adr_recog_weights)     # [max_predictions_per_seq_ar, ]

    features["masked_sr_positions"] = create_int_feature(masked_sr_positions)   # [max_predictions_per_seq_sr, ]
    features["masked_sr_labels"] = create_int_feature(masked_sr_labels)         # [max_predictions_per_seq_sr, ]
    features["masked_sr_weights"] = create_float_feature(masked_sr_weights)     # [max_predictions_per_seq_sr, ]

    features["pointer_cd_positions_spk1"] = create_int_feature(pointer_cd_positions_spk1)
    features["pointer_cd_positions_adr1"] = create_int_feature(pointer_cd_positions_adr1)
    features["pointer_cd_positions_spk2"] = create_int_feature(pointer_cd_positions_spk2)
    features["pointer_cd_positions_adr2"] = create_int_feature(pointer_cd_positions_adr2)
    features["pointer_cd_positions_spk3"] = create_int_feature(pointer_cd_positions_spk3)
    features["pointer_cd_positions_adr3"] = create_int_feature(pointer_cd_positions_adr3)
    features["pointer_cd_weights"] = create_float_feature(pointer_cd_weights)

    features["input_ids_msur"] = create_int_feature(input_ids_msur)
    features["input_mask_msur"] = create_int_feature(input_mask_msur)
    features["segment_ids_msur"] = create_int_feature(segment_ids_msur)
    features["speaker_ids_msur"] = create_int_feature(speaker_ids_msur)
    features["masked_sur_positions"] = create_int_feature(masked_sur_positions) # [max_utr_length, ]
    features["masked_sur_ids"] = create_int_feature(masked_sur_ids)             # [max_utr_length, ]
    features["masked_sur_weights"] = create_float_feature(masked_sur_weights)   # [max_utr_length, ]

    features["input_ids_snd"] = create_int_feature(input_ids_snd)
    features["input_mask_snd"] = create_int_feature(input_mask_snd)
    features["segment_ids_snd"] = create_int_feature(segment_ids_snd)
    features["speaker_ids_snd"] = create_int_feature(speaker_ids_snd)
    features["next_thread_labels"] = create_int_feature([next_thread_label])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))

    writers[writer_index].write(tf_example.SerializeToString())
    writer_index = (writer_index + 1) % len(writers)

    total_written += 1

    if inst_index < 10:
      tf.logging.info("*** Example ***")
      # tf.logging.info("tokens: %s" % " ".join(
      #     [tokenization.printable_text(x) for x in instance.tokens_nsp]))

      for feature_name in features.keys():
        feature = features[feature_name]
        values = []
        if feature.int64_list.value:
          values = feature.int64_list.value
        elif feature.float_list.value:
          values = feature.float_list.value
        tf.logging.info(
            "%s: %s" % (feature_name, " ".join([str(x) for x in values])))

  for writer in writers:
    writer.close()

  tf.logging.info("Wrote %d total instances", total_written)


def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature

def create_float_feature(values):
  feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
  return feature

  
def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  print_configuration_op(FLAGS)

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  # 1. load data
  """
  ctx and rsp are short for context and response  respectively
  spk and adr are short for speaker and addressee respectively
  """
  ctx_list = []
  ctx_spk_list = []
  ctx_adr_list = []
  ctx_relation_list = []
  rsp_list = []
  rsp_spk_list = []
  rsp_adr_list = []
  rsp_relation_list = []
  """
  One can increase the amount of pre-training data to further improve the performance. 
  In our experiments, the training set of the dateset used in Hu et al. (2019) was employed for pre-training.
  """
  for train_file in [FLAGS.train_file]:
    with open(train_file, 'r') as f:
      for line in f:
        data = json.loads(line)
        ctx_list.append(data['context'])
        ctx_spk_list.append(data['ctx_spk'])
        ctx_adr_list.append(data['ctx_adr'])
        ctx_relation_list.append(data['relation_at'])
        rsp_list.append(data['answer'])
        rsp_spk_list.append(data['ans_spk'])
        rsp_adr_list.append(data['ans_adr'])
        rsp_relation_list.append(data['ans_idx'])
    tf.logging.info("Reading from input file {} --> {} context-response pairs".format(train_file, len(ctx_list)))

  
  # 2. create training instances
  rng = random.Random(FLAGS.random_seed)
  instances = create_training_instances(
      ctx_list, ctx_spk_list, ctx_adr_list, ctx_relation_list, rsp_list, rsp_spk_list, rsp_adr_list, rsp_relation_list, 
      tokenizer, FLAGS.max_seq_length, FLAGS.max_utr_length, FLAGS.dupe_factor, FLAGS.masked_lm_prob, 
      FLAGS.max_predictions_per_seq, FLAGS.max_predictions_per_seq_ar, FLAGS.max_predictions_per_seq_sr, FLAGS.max_predictions_per_seq_cd, rng)

  
  # 3. write instance to example files
  output_files = [FLAGS.output_file]
  write_instance_to_example_files(
      instances, tokenizer, FLAGS.max_seq_length, FLAGS.max_utr_length, FLAGS.max_utr_num, 
      FLAGS.max_predictions_per_seq, FLAGS.max_predictions_per_seq_ar, FLAGS.max_predictions_per_seq_sr, FLAGS.max_predictions_per_seq_cd, output_files)


if __name__ == "__main__":
  # flags.mark_flag_as_required("train_file")
  # flags.mark_flag_as_required("output_file")
  # flags.mark_flag_as_required("vocab_file")
  tf.app.run()

