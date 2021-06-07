
CUDA_VISIBLE_DEVICES=0 python -u ../run_pretraining.py \
  --task_name MPC-BERT-pretraining \
  --input_file ../data/pretraining_data.tfrecord \
  --output_dir ../uncased_L-12_H-768_A-12_MPCBERT \
  --vocab_file ../uncased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file ../uncased_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint ../uncased_L-12_H-768_A-12/bert_model.ckpt \
  --max_seq_length 230 \
  --max_utr_length 30 \
  --max_utr_num 7 \
  --max_predictions_per_seq 25 \
  --max_predictions_per_seq_ar 4 \
  --max_predictions_per_seq_sr 2 \
  --max_predictions_per_seq_cd 2 \
  --train_batch_size 4 \
  --learning_rate 5e-5 \
  --mid_save_step 20000 \
  --num_train_epochs 1 \
  --warmup_proportion 0.1 > log_pretraining_MPCBERT.txt 2>&1 &
