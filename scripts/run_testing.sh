
CUDA_VISIBLE_DEVICES=0 python -u ../run_testing_ar.py \
    --test_dir ../data/ijcai2019/test_ar.tfrecord \
    --vocab_file ../uncased_L-12_H-768_A-12/vocab.txt \
    --bert_config_file ../uncased_L-12_H-768_A-12/bert_config.json \
    --max_seq_length 230 \
    --max_utr_num 7 \
    --eval_batch_size 256 \
    --restore_model_dir ../output/ijcai2019/PATH_TO_TEST_MODEL > log_testing_MPCBERT_ar.txt 2>&1 &
