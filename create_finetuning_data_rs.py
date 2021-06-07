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


def load_dataset(fname, n_negative):
    ctx_list = []
    ctx_spk_list = []
    rsp_list = []
    rsp_spk_list = []
    with open(fname, 'r') as f:
        for line in f:
            data = json.loads(line)
            ctx_list.append(data['context'])
            ctx_spk_list.append(data['ctx_spk'])
            rsp_list.append(data['answer'])
            rsp_spk_list.append(data['ans_spk'])
    print("matched context-response pairs: {}".format(len(ctx_list)))

    dataset = []
    index_list = list(range(len(ctx_list)))
    for i in range(len(ctx_list)):
        ctx = ctx_list[i]
        ctx_spk = ctx_spk_list[i]

        # positive
        rsp = rsp_list[i]
        rsp_spk = rsp_spk_list[i]
        dataset.append((i, ctx, ctx_spk, i, rsp, rsp_spk, 'follow'))

        # negative
        negatives = random.sample(index_list, n_negative)
        while i in negatives:
            negatives = random.sample(index_list, n_negative)
        assert i not in negatives
        for n_id in negatives:
            dataset.append((i, ctx, ctx_spk, n_id, rsp_list[n_id], rsp_spk, 'unfollow'))

    print("dataset_size: {}".format(len(dataset)))
    return dataset


class InputExample(object):
    def __init__(self, guid, ctx_id, ctx, ctx_spk, rsp_id, rsp, rsp_spk, label):
        """Constructs a InputExample."""
        self.guid = guid
        self.ctx_id = ctx_id
        self.ctx = ctx
        self.ctx_spk = ctx_spk
        self.rsp_id = rsp_id
        self.rsp = rsp
        self.rsp_spk = rsp_spk
        self.label = label


def create_examples(lines, set_type):
    """Creates examples for datasets."""
    examples = []
    for (i, line) in enumerate(lines):
        guid = "%s-%s" % (set_type, str(i))
        ctx_id = line[0]
        ctx = [tokenization.convert_to_unicode(utr) for utr in line[1]]
        ctx_spk = line[2]
        rsp_id = line[3]
        rsp = tokenization.convert_to_unicode(line[4])
        rsp_spk = line[5]
        label = tokenization.convert_to_unicode(line[-1])
        examples.append(InputExample(guid=guid, ctx_id=ctx_id, ctx=ctx, ctx_spk=ctx_spk, 
                                                rsp_id=rsp_id, rsp=rsp, rsp_spk=rsp_spk, label=label))
    return examples


def truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break

        if len(tokens_a) > len(tokens_b):
            trunc_tokens = tokens_a
        else:
            trunc_tokens = tokens_b

        if random.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, ctx_id, rsp_id, input_sents, input_mask, segment_ids, speaker_ids, label_id):
        self.ctx_id = ctx_id
        self.rsp_id = rsp_id
        self.input_sents = input_sents
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.speaker_ids = speaker_ids
        self.label_id = label_id


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {}
    for (i, label) in enumerate(label_list):  # ['0', '1']
        label_map[label] = i

    features = []
    for example in tqdm(examples, total=len(examples)):
        ctx_id = int(example.ctx_id)
        rsp_id = int(example.rsp_id)

        ctx_tokens = []
        ctx_spk = []
        for i in range(len(example.ctx)):
            utr = example.ctx[i]
            utr_spk = example.ctx_spk[i]
            utr_tokens = tokenizer.tokenize(utr)
            ctx_tokens.extend(utr_tokens)
            ctx_spk.extend([utr_spk]*len(utr_tokens))
        assert len(ctx_tokens) == len(ctx_spk)

        rsp_tokens = tokenizer.tokenize(example.rsp)

        # Account for [CLS], [SEP], [SEP] with "- 3"
        truncate_seq_pair(ctx_tokens, rsp_tokens, max_seq_length - 3)


        tokens = []
        segment_ids = []
        speaker_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        speaker_ids.append(0)
        for token_idx, token in enumerate(ctx_tokens):
            tokens.append(token)
            segment_ids.append(0)
            speaker_ids.append(ctx_spk[token_idx])
        tokens.append("[SEP]")
        segment_ids.append(0)
        speaker_ids.append(0)

        for token_idx, token in enumerate(rsp_tokens):
            tokens.append(token)
            segment_ids.append(1)
            # speaker_ids.append(0)  # 0 for mask speaker
            speaker_ids.append(example.rsp_spk)
        tokens.append("[SEP]")
        segment_ids.append(1)
        speaker_ids.append(example.rsp_spk)

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

        label_id = label_map[example.label]

        features.append(
            InputFeatures(
                ctx_id=ctx_id,
                rsp_id = rsp_id,
                input_sents=input_sents,
                input_mask=input_mask,
                segment_ids=segment_ids,
                speaker_ids=speaker_ids,
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
        features["ctx_id"] = create_int_feature([instance.ctx_id])
        features["rsp_id"] = create_int_feature([instance.rsp_id])
        features["input_sents"] = create_int_feature(instance.input_sents)
        features["input_mask"] = create_int_feature(instance.input_mask)
        features["segment_ids"] = create_int_feature(instance.segment_ids)
        features["speaker_ids"] = create_int_feature(instance.speaker_ids)
        features["label_ids"] = create_float_feature([instance.label_id])

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
    label_list = ["unfollow", "follow"]

    filenames = [FLAGS.train_file, FLAGS.valid_file, FLAGS.test_file]
    filetypes = ["train", "valid", "test"]
    file_n_negative = [1, 9, 9]

    for (filename, filetype, n_negative) in zip(filenames, filetypes, file_n_negative):
        dataset = load_dataset(filename, n_negative)
        examples = create_examples(dataset, filetype)
        features = convert_examples_to_features(examples, label_list, FLAGS.max_seq_length, tokenizer)
        new_filename = filename[:-5] + "_rs.tfrecord"
        write_instance_to_example_files(features, [new_filename])
        print('Convert {} to {} done'.format(filename, new_filename))

    print("Sub-process(es) done.")
