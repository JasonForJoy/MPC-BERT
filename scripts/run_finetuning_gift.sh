
CUDA_VISIBLE_DEVICES=0 python -u ../run_finetuning_ar_gift.py \
  --task_name fine_tuning \
  --train_dir ../data/ijcai2019/train_ar_gift.tfrecord \
  --valid_dir ../data/ijcai2019/dev_ar_gift.tfrecord \
  --output_dir ../output/ijcai2019 \
  --do_lower_case True \
  --vocab_file ../uncased_L-12_H-768_A-12_MPCBERT/vocab.txt \
  --bert_config_file ../uncased_L-12_H-768_A-12_MPCBERT/bert_config.json \
  --init_checkpoint ../uncased_L-12_H-768_A-12_MPCBERT/bert_model.ckpt \
  --max_seq_length 230 \
  --max_utr_num 7 \
  --do_train True  \
  --train_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --warmup_proportion 0.1 > log_finetuning_MPCBERT_ar_GIFT.txt 2>&1 &
