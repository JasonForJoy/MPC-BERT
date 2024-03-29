# coding=utf-8
import json
import random
import numpy as np
import collections
from tqdm import tqdm
import tokenization
import tensorflow as tf


""" Hu et al. GSN: A Graph-Structured Network for Multi-Party Dialogues. IJCAI 2019. """
tf.flags.DEFINE_string("train_file", "./data/ijcai2019/train.json", 
                       "path to train file")
tf.flags.DEFINE_string("valid_file", "./data/ijcai2019/dev.json", 
                       "path to valid file")
tf.flags.DEFINE_string("test_file", "./data/ijcai2019/test.json", 
                       "path to test file")
tf.flags.DEFINE_integer("max_seq_length", 230, 
                        "max sequence length of concatenated context and response")
tf.flags.DEFINE_integer("max_utr_num", 7, 
                        "Maximum utterance number.")

""" 
Ouchi et al. Addressee and Response Selection for Multi-Party Conversation. EMNLP 2016.
relesed the original dataset which is composed of 3 experimental settings according to conversation lengths.

In our experiments, we used the version processed and used in 
Le et al. Who Is Speaking to Whom? Learning to Identify Utterance Addressee in Multi-Party Conversations. EMNLP 2019. 
"""

# Length-5
# tf.flags.DEFINE_string("train_file", "./data/emnlp2016/5_train.json", 
#                        "path to train file")
# tf.flags.DEFINE_string("valid_file", "./data/emnlp2016/5_dev.json", 
#                        "path to valid file")
# tf.flags.DEFINE_string("test_file", "./data/emnlp2016/5_test.json", 
#                        "path to test file")
# tf.flags.DEFINE_integer("max_seq_length", 120, 
#                         "max sequence length of concatenated context and response")
# tf.flags.DEFINE_integer("max_utr_num", 5, 
#                         "Maximum utterance number.")

# Length-10
# tf.flags.DEFINE_string("train_file", "./data/emnlp2016/10_train.json", 
#                        "path to train file")
# tf.flags.DEFINE_string("valid_file", "./data/emnlp2016/10_dev.json", 
#                        "path to valid file")
# tf.flags.DEFINE_string("test_file", "./data/emnlp2016/10_test.json", 
#                        "path to test file")
# tf.flags.DEFINE_integer("max_seq_length", 220, 
#                         "max sequence length of concatenated context and response")
# tf.flags.DEFINE_integer("max_utr_num", 10, 
#                         "Maximum utterance number.")

# Length-15
# tf.flags.DEFINE_string("train_file", "./data/emnlp2016/15_train.json", 
#                        "path to train file")
# tf.flags.DEFINE_string("valid_file", "./data/emnlp2016/15_dev.json", 
#                        "path to valid file")
# tf.flags.DEFINE_string("test_file", "./data/emnlp2016/15_test.json", 
#                        "path to test file")
# tf.flags.DEFINE_integer("max_seq_length", 320, 
#                         "max sequence length of concatenated context and response")
# tf.flags.DEFINE_integer("max_utr_num", 15, 
#                         "Maximum utterance number.")

tf.flags.DEFINE_string("vocab_file", "./uncased_L-12_H-768_A-12/vocab.txt", 
                       "path to vocab file")
tf.flags.DEFINE_bool("do_lower_case", True,
                     "whether to lower case the input text")



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


def load_dataset(fname):
    dataset = []
    with open(fname, 'r') as f:
        for line in f:
            data = json.loads(line)
            ctx = data['context']
            ctx_spk = data['ctx_spk']
            ctx_adr = data['ctx_adr']
            rsp = data['answer']
            rsp_spk = data['ans_spk']
            rsp_adr = data['ans_adr']
            ctx_relation = data['relation_at']
            rsp_relation = data['ans_idx']

            integrate_ctx = ctx + [rsp]
            integrate_ctx_spk = ctx_spk + [rsp_spk]
            integrate_ctx_adr = ctx_adr + [rsp_adr]
            assert len(integrate_ctx) == len(integrate_ctx_spk)
            assert len(integrate_ctx) == len(integrate_ctx_adr)
            # integrate_ctx_relation = ctx_relation + [[len(ctx), rsp_relation]]  # avoid information leakage of response addreessee
            
            # test only the last utterance
            label = [0 for _ in range(len(ctx))]
            for cand_utr_id_spk, cand_utr_spk in enumerate(ctx_spk):  # consider only the preceding utterances
                if cand_utr_spk == rsp_adr:
                    label[cand_utr_id_spk] = 1
            assert sum(label) > 0
            
            # construct the reply mask
            reply_mask = [[0 for _ in range(len(integrate_ctx))] for _ in range(len(integrate_ctx))]
            for relation in ctx_relation:
                tgt, src = relation
                reply_mask[tgt][src] = 1  # reply
                reply_mask[src][tgt] = 2  # replied_by
            for diagonal in range(len(integrate_ctx)):
                reply_mask[diagonal][diagonal] = 3  # reply_to_itself

            dataset.append((ctx, ctx_spk, rsp, rsp_spk, reply_mask, label))

    print("dataset_size: {}".format(len(dataset)))
    return dataset


class InputExample(object):
    def __init__(self, guid, ctx, ctx_spk, rsp, rsp_spk, reply_mask, label):
        """Constructs a InputExample."""
        self.guid = guid
        self.ctx = ctx
        self.ctx_spk = ctx_spk
        self.rsp = rsp
        self.rsp_spk = rsp_spk
        self.reply_mask = reply_mask
        self.label = label


def create_examples(lines, set_type):
    """Creates examples for datasets."""
    examples = []
    for (i, line) in enumerate(lines):
        guid = "%s-%s" % (set_type, str(i))
        ctx = [tokenization.convert_to_unicode(utr) for utr in line[0]]
        ctx_spk = line[1]
        rsp = tokenization.convert_to_unicode(line[2])
        rsp_spk = line[3]
        reply_mask = line[4]
        label = line[-1]
        examples.append(InputExample(guid=guid, ctx=ctx, ctx_spk=ctx_spk, rsp=rsp, rsp_spk=rsp_spk, reply_mask=reply_mask, label=label))
    return examples


def truncate_seq_pair(ctx_tokens, rsp_tokens, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        utr_lens = [len(utr_tokens) for utr_tokens in ctx_tokens]
        total_length = sum(utr_lens) + len(rsp_tokens)
        if total_length <= max_length:
            break

        # truncate the longest utterance or response
        if sum(utr_lens) > len(rsp_tokens):
            trunc_tokens = ctx_tokens[np.argmax(np.array(utr_lens))]
        else:
            trunc_tokens = rsp_tokens
        assert len(trunc_tokens) >= 1

        if random.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_sents, input_mask, segment_ids, speaker_ids, cls_positions, rsp_position, reply_mask_utr2word_flatten, utr_lens, label_id):
        self.input_sents = input_sents
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.speaker_ids = speaker_ids
        self.cls_positions = cls_positions
        self.rsp_position = rsp_position
        self.reply_mask_utr2word_flatten = reply_mask_utr2word_flatten
        self.utr_lens = utr_lens
        self.label_id = label_id


def convert_examples_to_features(examples, max_seq_length, max_utr_num, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for example in tqdm(examples, total=len(examples)):

        ctx_tokens = []
        for utr in example.ctx:
            utr_tokens = tokenizer.tokenize(utr)
            ctx_tokens.append(utr_tokens)
        assert len(ctx_tokens) == len(example.ctx_spk)

        rsp_tokens = tokenizer.tokenize(example.rsp)

        # [CLS]s for context, [CLS] for response, [SEP]
        max_num_tokens = max_seq_length - len(ctx_tokens) - 1 - 1
        truncate_seq_pair(ctx_tokens, rsp_tokens, max_num_tokens)

        tokens = []
        segment_ids = []
        speaker_ids = []
        cls_positions = []
        rsp_position = []
        reply_mask_utr2word = [[] for _ in range(len(example.reply_mask))]
        utr_lens = []

        # utterances
        for i in range(len(ctx_tokens)):
            utr_tokens = ctx_tokens[i]
            utr_spk = example.ctx_spk[i]

            utr_lens.append(len(utr_tokens) + 1)  # +1 for [CLS]
            cls_positions.append(len(tokens))
            tokens.append("[CLS]")
            segment_ids.append(0)
            speaker_ids.append(utr_spk)
            for j in range(len(example.reply_mask)):
                reply_mask_utr2word[j].append(example.reply_mask[j][i])

            for token in utr_tokens:
                tokens.append(token)
                segment_ids.append(0)
                speaker_ids.append(utr_spk)
                for j in range(len(example.reply_mask)):
                    reply_mask_utr2word[j].append(example.reply_mask[j][i])

        # response
        utr_lens.append(len(rsp_tokens) + 2)  # +2 for [CLS] and [SEP]
        rsp_position.append(len(cls_positions))
        cls_positions.append(len(tokens))
        tokens.append("[CLS]")
        segment_ids.append(0)
        speaker_ids.append(example.rsp_spk)
        for j in range(len(example.reply_mask)):
            reply_mask_utr2word[j].append(example.reply_mask[j][-1])

        for token in rsp_tokens:
            tokens.append(token)
            segment_ids.append(0)
            speaker_ids.append(example.rsp_spk)
            for j in range(len(example.reply_mask)):
                reply_mask_utr2word[j].append(example.reply_mask[j][-1])

        tokens.append("[SEP]")
        segment_ids.append(0)
        speaker_ids.append(example.rsp_spk)
        for j in range(len(example.reply_mask)):
            reply_mask_utr2word[j].append(example.reply_mask[j][-1])

        
        assert len(utr_lens) == len(reply_mask_utr2word)
        for i in range(len(reply_mask_utr2word)):
            assert len(reply_mask_utr2word[i]) <= max_seq_length
            while len(reply_mask_utr2word[i]) < max_seq_length:
                reply_mask_utr2word[i].append(0)
            assert len(reply_mask_utr2word[i]) == max_seq_length

        assert len(reply_mask_utr2word) <= max_utr_num
        while len(reply_mask_utr2word) < max_utr_num:
            reply_mask_utr2word.append([0]*max_seq_length)
        assert len(reply_mask_utr2word) == max_utr_num

        reply_mask_utr2word_flatten = []
        for x in reply_mask_utr2word:
            reply_mask_utr2word_flatten.extend(x)
        assert len(reply_mask_utr2word_flatten) == max_seq_length * max_utr_num

        assert len(utr_lens) <= max_utr_num
        while len(utr_lens) < max_utr_num:
            utr_lens.append(0)
        assert len(utr_lens) == max_utr_num
        
        input_sents = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_sents)
        assert len(input_sents) <= max_seq_length
        while len(input_sents) < max_seq_length:
            input_sents.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            speaker_ids.append(0)
        assert len(input_sents) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(speaker_ids) == max_seq_length

        assert len(cls_positions) <= max_utr_num
        while len(cls_positions) < max_utr_num:
            cls_positions.append(0)
        assert len(cls_positions) == max_utr_num

        # test only the last utterance
        label_id = example.label
        assert len(label_id) <= max_utr_num
        while len(label_id) < max_utr_num:
            label_id.append(0)
        assert len(label_id) == max_utr_num

        assert len(label_id) <= max_utr_num
        while len(label_id) < max_utr_num:
            label_id.append([0] * max_utr_num)
        assert len(label_id) == max_utr_num

        features.append(
            InputFeatures(
                input_sents=input_sents,
                input_mask=input_mask,
                segment_ids=segment_ids,
                speaker_ids=speaker_ids,
                cls_positions=cls_positions,
                rsp_position=rsp_position,
                reply_mask_utr2word_flatten=reply_mask_utr2word_flatten,
                utr_lens=utr_lens,
                label_id=label_id))

    return features


def write_instance_to_example_files(instances, output_files):
    writers = []

    for output_file in output_files:
        writers.append(tf.python_io.TFRecordWriter(output_file))

    writer_index = 0
    total_written = 0
    for (inst_index, instance) in enumerate(instances):
        features = collections.OrderedDict()
        features["input_sents"] = create_int_feature(instance.input_sents)
        features["input_mask"] = create_int_feature(instance.input_mask)
        features["segment_ids"] = create_int_feature(instance.segment_ids)
        features["speaker_ids"] = create_int_feature(instance.speaker_ids)
        features["cls_positions"] = create_int_feature(instance.cls_positions)
        features["rsp_position"] = create_int_feature(instance.rsp_position)
        features["reply_mask_utr2word_flatten"] = create_int_feature(instance.reply_mask_utr2word_flatten)
        features["utr_lens"] = create_int_feature(instance.utr_lens)
        features["label_ids"] = create_int_feature(instance.label_id)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))

        writers[writer_index].write(tf_example.SerializeToString())
        writer_index = (writer_index + 1) % len(writers)

        total_written += 1

    print("write_{}_instance_to_example_files".format(total_written))

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


def create_int_feature(values):
	feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
	return feature

def create_float_feature(values):
	feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
	return feature



if __name__ == "__main__":

    FLAGS = tf.flags.FLAGS
    print_configuration_op(FLAGS)

    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    filenames = [FLAGS.train_file, FLAGS.valid_file, FLAGS.test_file]
    filetypes = ["train", "valid", "test"]
    for (filename, filetype) in zip(filenames, filetypes):
        dataset = load_dataset(filename)
        examples = create_examples(dataset, filetype)
        features = convert_examples_to_features(examples, FLAGS.max_seq_length, FLAGS.max_utr_num, tokenizer)
        new_filename = filename[:-5] + "_ar_gift.tfrecord"
        write_instance_to_example_files(features, [new_filename])
        print('Convert {} to {} done'.format(filename, new_filename))

    print("Sub-process(es) done.")
