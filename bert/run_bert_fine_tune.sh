set -x
base_dir=/home/zfgao/work/Fine-grained-decomposition/bert
data_dir_base=/mnt/zfgao/data/nlp_data/GLUE
check_point_dir=/mnt/zfgao/checkpoint/sshlab/bert

export WANDB_PROJECT=BERT-base-MPO-analysis
export OMP_NUM_THREADS=16
echo $gpu_num
function run_task() {
  export CUDA_VISIBLE_DEVICES=$1
  COMMON_ARGS="--data_dir="$data_dir_base/$2" --model_name_or_path=${14} --tokenizer_name=bert-base-uncased --evaluation_strategy=steps --do_train --do_eval --logging_steps=50 --overwrite_output_dir --do_train --eval_steps=500 --save_steps=50000 --gpu_num=$1 --task_name=$2 --warmup_step=$3 --run_name=$7 --learning_rate=$4 --num_train_epochs=$5 --per_device_train_batch_size=$6 --output_dir="$check_point_dir/$s7" --max_seq_length=$8 --mpo_lr=$9 --mpo_layers=${10} --emb_trunc=${11} --linear_trunc=${12} --attention_trunc=${13} --load_layer=${15} ${16}"
  nohup python $base_dir/run_glue_bert.py \
      ${COMMON_ARGS} \
      --do_train --do_eval > $base_dir/log_analysis/BERT_Base/Top-K/$7.log 2>&1 &
}
# run_task 8 SST-2 500 2e-5 3.0 32 sst_baseline_test 128 2.8e-6 nompo 1000 1000 1000 bert-base-uncased Noload noupdate --do_train\ --metric_for_best_model="acc"\ --eval_steps=500
# run_task 8 MRPC 500 2e-5 3.0 32 mprc_base_top5 128 2.8e-6 nompo 1000 1000 1000 bert-base-uncased Noload noupdate --do_train\ --metric_for_best_model="acc"\ --eval_steps=500\ --mode=ChildTuning-D\ --topN=6
# run_task 4 CoLA 500 1e-5 20.0 16 cola_base_top9_1 128 1e-5 nompo 10000 10000 10000 bert-base-uncased Noload --mode=ChildTuning-D\ --topN=9
# run_task 5 CoLA 500 2e-5 20.0 16 cola_base_top9_2 128 7e-6 nompo 10000 10000 10000 bert-base-uncased Noload --mode=ChildTuning-D\ --topN=9
# run_task 6 CoLA 500 3e-5 20.0 16 cola_base_top9_3 128 5e-6 nompo 10000 10000 10000 bert-base-uncased Noload --mode=ChildTuning-D\ --topN=9

# run_task 4 CoLA 500 2e-5 20.0 16 cola_base_mpo4 128 2.8e-6 word_embed,FFN_1,FFN_2,attention 10000 10000 10000 bert-base-uncased Noload --do_train
# run_task 5 CoLA 500 2e-5 20.0 16 cola_base_mpo5 128 2.8e-6 FFN_1,FFN_2,attention 10000 10000 10000 bert-base-uncased Noload --do_train
# run_task 6 CoLA 500 2e-5 20.0 16 cola_base_mpo6 128 2.8e-6 FFN_1,FFN_2 10000 10000 10000 bert-base-uncased Noload --do_train

# Dynamic MPO BERT-base
# run_task 1 STS-B 500 2e-5 10.0 16 stsb_mpo_top9_dynamic800 128 2.8e-6 nompo 10000 10000 10000 bert-base-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=9\ --dynamic_decom\ --split_num=3
# run_task 2 MNLI 500 2e-5 10.0 16 mnli_mpo_top9_dynamic800 128 2.8e-6 nompo 10000 10000 10000 bert-base-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=9\ --dynamic_decom\ --split_num=3
# run_task 3 QNLI 500 2e-5 10.0 16 qnli_mpo_top9_dynamic800 128 2.8e-6 nompo 10000 10000 10000 bert-base-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=9\ --dynamic_decom\ --split_num=3
# run_task 4 QQP 500 2e-5 10.0 16 qqp_mpo_top9_dynamic800 128 2.8e-6 nompo 10000 10000 10000 bert-base-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=9\ --dynamic_decom\ --split_num=3
# run_task 5 SST-2 500 2e-5 10.0 16 sst2_mpo_top9_dynamic800_3 128 2.8e-6 nompo 10000 10000 10000 bert-base-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=9\ --dynamic_decom\ --split_num=3
# run_task 6 CoLA 500 2e-5 10.0 16 cola_mpo_top9_dynamic800 128 2.8e-6 nompo 10000 10000 10000 bert-base-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=9\ --dynamic_decom\ --split_num=3
# run_task 1 RTE 500 2e-5 20.0 16 rte_mpo_top9_dynamic800 128 2.8e-6 nompo 10000 10000 10000 bert-base-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=9\ --dynamic_decom\ --split_num=3
# Dynamic MPO BERT-large
# run_task 2 MNLI 500 2e-5 10.0 16 mnli_mpo_top9_dynamic 128 2.8e-6 nompo 10000 10000 10000 bert-large-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=9\ --dynamic_decom\ --split_num=3
# run_task 1 SST-2 500 2e-5 10.0 16 sst2_mpo_top9_dynamic 128 2.8e-6 nompo 10000 10000 10000 bert-large-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=9\ --dynamic_decom\ --split_num=3
# run_task 5 QQP 500 2e-5 10.0 16 qqp_mpo_top9_dynamic 128 2.8e-6 nompo 10000 10000 10000 bert-large-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=9\ --dynamic_decom\ --split_num=3
# run_task 6 MRPC 500 2e-5 10.0 16 mrpc_mpo_top9_dynamic 128 2.8e-6 nompo 10000 10000 10000 bert-large-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=9\ --dynamic_decom\ --split_num=3
# run_task 7 CoLA 500 2e-5 10.0 16 cola_mpo_top9_dynamic 128 2.8e-6 nompo 10000 10000 10000 bert-large-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=9\ --dynamic_decom\ --split_num=3

# Static MPO
# run_task 4 MNLI 500 2e-5 10.0 16 mnli_mpo_top9_static 128 2.8e-6 nompo 10000 10000 10000 bert-base-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=9
# run_task 5 QNLI 500 2e-5 10.0 16 qnli_mpo_top9_static 128 2.8e-6 nompo 10000 10000 10000 bert-base-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=9
# run_task 6 STS-B 500 2e-5 10.0 16 stsb_mpo_top9_static 128 2.8e-6 nompo 10000 10000 10000 bert-base-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=9
# run_task 7 RTE 500 2e-5 10.0 16 rte_mpo_top9_static 128 2.8e-6 nompo 10000 10000 10000 bert-base-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=9
# run_task 8 QQP 500 2e-5 10.0 16 qqp_mpo_top9_static 128 2.8e-6 nompo 10000 10000 10000 bert-base-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=9


# Static MPO bert-large
# run_task 1 MRPC 500 2e-5 10.0 16 mrpc_mpo_top9_static 128 2.8e-6 nompo 10000 10000 10000 bert-large-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=9
# run_task 2 MNLI 500 2e-5 10.0 16 mnli_mpo_top9_static 128 2.8e-6 nompo 10000 10000 10000 bert-large-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=9
# run_task 4 QNLI 500 2e-5 10.0 16 qnli_mpo_top9_static 128 2.8e-6 nompo 10000 10000 10000 bert-large-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=9
# run_task 5 STS-B 500 2e-5 10.0 16 stsb_mpo_top9_static 128 2.8e-6 nompo 10000 10000 10000 bert-large-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=9
# run_task 6 RTE 500 2e-5 10.0 16 rte_mpo_top9_static 128 2.8e-6 nompo 10000 10000 10000 bert-large-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=9
# run_task 7 QQP 500 2e-5 10.0 16 qqp_mpo_top9_static 128 2.8e-6 nompo 10000 10000 10000 bert-large-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=9
# run_task 7 CoLA 500 2e-5 10.0 16 cola_mpo_top9_static 128 2.8e-6 nompo 10000 10000 10000 bert-large-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=9
# run_task 7 SST-2 500 2e-5 10.0 16 sst2_mpo_top9_static 128 2.8e-6 nompo 10000 10000 10000 bert-large-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=9



# BERT-tiny
# run_task 3 MRPC 500 2e-5 10.0 16 bert_tiny_mrpc 128 2.8e-6 nompo 10000 10000 10000 prajjwal1/bert-tiny Noload --do_train


# BERT-base-SVD
# run_task 4 MNLI 500 2e-5 10.0 16 mnli_svd_top9_static 128 2.8e-6 nompo 10000 10000 10000 bert-base-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=9
# run_task 5 QNLI 500 2e-5 10.0 16 qnli_svd_top9_static 128 2.8e-6 nompo 10000 10000 10000 bert-base-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=9
# run_task 6 STS-B 500 2e-5 10.0 16 stsb_svd_top9_static 128 2.8e-6 nompo 10000 10000 10000 bert-base-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=9
# run_task 3 RTE 500 2e-5 10.0 16 rte_svd_top9_static 128 2.8e-6 nompo 10000 10000 10000 bert-base-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=9
# run_task 8 QQP 500 2e-5 10.0 16 qqp_svd_top9_static 128 2.8e-6 nompo 10000 10000 10000 bert-base-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=9
# run_task 7 SST-2 500 2e-5 10.0 16 sst2_svd_top9_static 128 2.8e-6 nompo 10000 10000 10000 bert-base-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=9
# run_task 1 MRPC 500 2e-5 10.0 16 mrpc_svd_top9_static 128 2.8e-6 nompo 10000 10000 10000 bert-base-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=9
# run_task 2 CoLA 500 2e-5 10.0 16 cola_svd_top9_static 128 2.8e-6 nompo 10000 10000 10000 bert-base-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=9


# BERT-large-SVD
# run_task 4 MNLI 500 2e-5 5.0 16 mnli_svd_top9_static 128 2.8e-6 nompo 10000 10000 10000 bert-large-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=9
# run_task 5 QNLI 500 2e-5 5.0 16 qnli_svd_top9_static 128 2.8e-6 nompo 10000 10000 10000 bert-large-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=9
# run_task 8 STS-B 500 2e-5 5.0 16 stsb_svd_top9_static 128 2.8e-6 nompo 10000 10000 10000 bert-large-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=9
# run_task 3 RTE 500 2e-5 5.0 16 rte_svd_top9_static 128 2.8e-6 nompo 10000 10000 10000 bert-large-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=9
# run_task 8 QQP 500 2e-5 5.0 16 qqp_svd_top9_static 128 2.8e-6 nompo 10000 10000 10000 bert-large-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=9
# run_task 0 SST-2 500 2e-5 5.0 16 sst2_svd_top9_static 128 2.8e-6 nompo 10000 10000 10000 bert-large-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=9
# run_task 1 MRPC 500 2e-5 5.0 16 mrpc_svd_top9_static 128 2.8e-6 nompo 10000 10000 10000 bert-large-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=9
# run_task 2 CoLA 500 2e-5 5.0 16 cola_svd_top9_static 128 2.8e-6 nompo 10000 10000 10000 bert-large-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=9

# BERT-small-svd
# run_task 1 CoLA 500 2e-5 10.0 16 cola_bert_small_svd 128 2.8e-6 nompo 10000 10000 10000 prajjwal1/bert-small Noload --do_train\ --mode=ChildTuning-D\ --topN=4
# run_task 1 RTE 500 2e-5 10.0 16 rte_bert_small_svd 128 2.8e-6 nompo 10000 10000 10000 prajjwal1/bert-small Noload --do_train\ --mode=ChildTuning-D\ --topN=4
# run_task 1 MNLI 500 2e-5 10.0 16 mnli_bert_small_svd 128 2.8e-6 nompo 10000 10000 10000 prajjwal1/bert-small Noload --do_train\ --mode=ChildTuning-D\ --topN=4
# run_task 2 QNLI 500 2e-5 10.0 16 qnli_bert_small_svd 128 2.8e-6 nompo 10000 10000 10000 prajjwal1/bert-small Noload --do_train\ --mode=ChildTuning-D\ --topN=4
# run_task 2 SST-2 500 2e-5 10.0 16 sst2_bert_small_svd 128 2.8e-6 nompo 10000 10000 10000 prajjwal1/bert-small Noload --do_train\ --mode=ChildTuning-D\ --topN=4
# run_task 3 QQP 500 2e-5 10.0 16 qqp_bert_small_svd 128 2.8e-6 nompo 10000 10000 10000 prajjwal1/bert-small Noload --do_train\ --mode=ChildTuning-D\ --topN=4
# run_task 3 STS-B 500 2e-5 10.0 16 stsb_bert_small_svd 128 2.8e-6 nompo 10000 10000 10000 prajjwal1/bert-small Noload --do_train\ --mode=ChildTuning-D\ --topN=4


# # BERT-medium-svd
# run_task 0 CoLA 500 2e-5 5.0 16 cola_bert_medium_svd 128 2.8e-6 nompo 10000 10000 10000 prajjwal1/bert-medium Noload --do_train\ --mode=ChildTuning-D\ --topN=4
# run_task 0 RTE 500 2e-5 5.0 16 rte_bert_medium_svd 128 2.8e-6 nompo 10000 10000 10000 prajjwal1/bert-medium Noload --do_train\ --mode=ChildTuning-D\ --topN=4
# run_task 0 MNLI 500 2e-5 5.0 16 mnli_bert_medium_svd 128 2.8e-6 nompo 10000 10000 10000 prajjwal1/bert-medium Noload --do_train\ --mode=ChildTuning-D\ --topN=4
# run_task 0 QNLI 500 2e-5 5.0 16 qnli_bert_medium_svd 128 2.8e-6 nompo 10000 10000 10000 prajjwal1/bert-medium Noload --do_train\ --mode=ChildTuning-D\ --topN=4
# run_task 1 SST-2 500 2e-5 5.0 16 sst2_bert_medium_svd 128 2.8e-6 nompo 10000 10000 10000 prajjwal1/bert-medium Noload --do_train\ --mode=ChildTuning-D\ --topN=4
# run_task 1 QQP 500 2e-5 5.0 16 qqp_bert_medium_svd 128 2.8e-6 nompo 10000 10000 10000 prajjwal1/bert-medium Noload --do_train\ --mode=ChildTuning-D\ --topN=4
# run_task 1 STS-B 500 2e-5 5.0 16 stsb_bert_medium_svd 128 2.8e-6 nompo 10000 10000 10000 prajjwal1/bert-medium Noload --do_train\ --mode=ChildTuning-D\ --topN=4
# run_task 1 MRPC 500 2e-5 5.0 16 mrpc_bert_medium_svd 128 2.8e-6 nompo 10000 10000 10000 prajjwal1/bert-medium Noload --do_train\ --mode=ChildTuning-D\ --topN=4


# BERT-small-MPO-static
# run_task 2 CoLA 500 2e-5 5.0 16 cola_bert_small_MPO2 128 2.8e-6 nompo 10000 10000 10000 prajjwal1/bert-small Noload --do_train\ --mode=ChildTuning-D\ --topN=4
# run_task 2 RTE 500 2e-5 5.0 16 rte_bert_small_MPO2 128 2.8e-6 nompo 10000 10000 10000 prajjwal1/bert-small Noload --do_train\ --mode=ChildTuning-D\ --topN=4
# run_task 2 MNLI 500 2e-5 5.0 16 mnli_bert_small_MPO2 128 2.8e-6 nompo 10000 10000 10000 prajjwal1/bert-small Noload --do_train\ --mode=ChildTuning-D\ --topN=4
# run_task 1 QNLI 500 2e-5 5.0 16 qnli_bert_small_MPO2 128 2.8e-6 nompo 10000 10000 10000 prajjwal1/bert-small Noload --do_train\ --mode=ChildTuning-D\ --topN=4
# run_task 0 SST-2 500 2e-5 5.0 16 sst2_bert_small_MPO2 128 2.8e-6 nompo 10000 10000 10000 prajjwal1/bert-small Noload --do_train\ --mode=ChildTuning-D\ --topN=4
# run_task 5 QQP 500 2e-5 5.0 16 qqp_bert_small_MPO2 128 2.8e-6 nompo 10000 10000 10000 prajjwal1/bert-small Noload --do_train\ --mode=ChildTuning-D\ --topN=4
# run_task 5 STS-B 500 2e-5 5.0 16 stsb_bert_small_MPO2 128 2.8e-6 nompo 10000 10000 10000 prajjwal1/bert-small Noload --do_train\ --mode=ChildTuning-D\ --topN=4
# run_task 5 MRPC 500 2e-5 5.0 16 mrpc_bert_small_MPO2 128 2.8e-6 nompo 10000 10000 10000 prajjwal1/bert-small Noload --do_train\ --mode=ChildTuning-D\ --topN=4

# BERT-small-MPO-dynamic
# run_task 0 CoLA 500 2e-5 10.0 16 cola_bert_small_MPO_dynamic 128 2.8e-6 nompo 10000 10000 10000 prajjwal1/bert-small Noload --do_train\ --mode=ChildTuning-D\ --topN=4\ --split_num=4
# run_task 0 RTE 500 2e-5 10.0 16 rte_bert_small_MPO_dynamic 128 2.8e-6 nompo 10000 10000 10000 prajjwal1/bert-small Noload --do_train\ --mode=ChildTuning-D\ --topN=4\ --split_num=4
# run_task 0 MNLI 500 2e-5 5.0 16 mnli_bert_small_MPO_dynamic 128 2.8e-6 nompo 10000 10000 10000 prajjwal1/bert-small Noload --do_train\ --mode=ChildTuning-D\ --topN=4\ --split_num=4
# run_task 1 QNLI 500 2e-5 10.0 16 qnli_bert_small_MPO_dynamic 128 2.8e-6 nompo 10000 10000 10000 prajjwal1/bert-small Noload --do_train\ --mode=ChildTuning-D\ --topN=4\ --split_num=4
# run_task 1 SST-2 500 2e-5 10.0 16 sst2_bert_small_MPO_dynamic 128 2.8e-6 nompo 10000 10000 10000 prajjwal1/bert-small Noload --do_train\ --mode=ChildTuning-D\ --topN=4\ --split_num=4
# run_task 2 QQP 500 2e-5 5.0 16 qqp_bert_small_MPO_dynamic 128 2.8e-6 nompo 10000 10000 10000 prajjwal1/bert-small Noload --do_train\ --mode=ChildTuning-D\ --topN=4\ --split_num=4
# run_task 2 STS-B 500 2e-5 10.0 16 stsb_bert_small_MPO_dynamic 128 2.8e-6 nompo 10000 10000 10000 prajjwal1/bert-small Noload --do_train\ --mode=ChildTuning-D\ --topN=4\ --split_num=4
# run_task 2 MRPC 500 2e-5 10.0 16 mrpc_bert_small_MPO_dynamic 128 2.8e-6 nompo 10000 10000 10000 prajjwal1/bert-small Noload --do_train\ --mode=ChildTuning-D\ --topN=4\ --split_num=4



# BERT-medium-MPO-static
# run_task 0 CoLA 500 2e-5 5.0 32 cola_bert_medium_MPO_static2 128 2.8e-6 nompo 10000 10000 10000 prajjwal1/bert-medium Noload --do_train\ --mode=ChildTuning-D\ --topN=6
# run_task 0 RTE 500 2e-5 5.0 32 rte_bert_medium_MPO_static2 128 2.8e-6 nompo 10000 10000 10000 prajjwal1/bert-medium Noload --do_train\ --mode=ChildTuning-D\ --topN=6
# run_task 0 MNLI 500 2e-5 5.0 32 mnli_bert_medium_MPO_static2 128 2.8e-6 nompo 10000 10000 10000 prajjwal1/bert-medium Noload --do_train\ --mode=ChildTuning-D\ --topN=6
# run_task 0 QNLI 500 2e-5 5.0 32 qnli_bert_medium_MPO_static2 128 2.8e-6 nompo 10000 10000 10000 prajjwal1/bert-medium Noload --do_train\ --mode=ChildTuning-D\ --topN=6
# run_task 5 SST-2 500 2e-5 5.0 32 sst2_bert_medium_MPO_static2 128 2.8e-6 nompo 10000 10000 10000 prajjwal1/bert-medium Noload --do_train\ --mode=ChildTuning-D\ --topN=6
# run_task 5 QQP 500 2e-5 5.0 32 qqp_bert_medium_MPO_static2 128 2.8e-6 nompo 10000 10000 10000 prajjwal1/bert-medium Noload --do_train\ --mode=ChildTuning-D\ --topN=6
# run_task 5 STS-B 500 2e-5 5.0 32 stsb_bert_medium_MPO_static2 128 2.8e-6 nompo 10000 10000 10000 prajjwal1/bert-medium Noload --do_train\ --mode=ChildTuning-D\ --topN=6
# run_task 5 MRPC 500 2e-5 5.0 32 mrpc_bert_medium_MPO_static2 128 2.8e-6 nompo 10000 10000 10000 prajjwal1/bert-medium Noload --do_train\ --mode=ChildTuning-D\ --topN=6

# BERT-small-MPO-dynamic
# run_task 0 CoLA 500 2e-5 10.0 16 cola_bert_medium_MPO_dynamic2 128 2.8e-6 nompo 10000 10000 10000 prajjwal1/bert-medium Noload --do_train\ --mode=ChildTuning-D\ --topN=6\ --split_num=3
# run_task 0 RTE 500 2e-5 10.0 16 rte_bert_medium_MPO_dynamic2 128 2.8e-6 nompo 10000 10000 10000 prajjwal1/bert-medium Noload --do_train\ --mode=ChildTuning-D\ --topN=6\ --split_num=3
# run_task 1 MNLI 500 2e-5 5.0 16 mnli_bert_medium_MPO_dynamic2 128 2.8e-6 nompo 10000 10000 10000 prajjwal1/bert-medium Noload --do_train\ --mode=ChildTuning-D\ --topN=6\ --split_num=3
# run_task 4 QNLI 500 2e-5 10.0 16 qnli_bert_medium_MPO_dynamic2 128 2.8e-6 nompo 10000 10000 10000 prajjwal1/bert-medium Noload --do_train\ --mode=ChildTuning-D\ --topN=6\ --split_num=3
# run_task 4 SST-2 500 2e-5 10.0 16 sst2_bert_medium_MPO_dynamic2 128 2.8e-6 nompo 10000 10000 10000 prajjwal1/bert-medium Noload --do_train\ --mode=ChildTuning-D\ --topN=6\ --split_num=3
# run_task 3 QQP 500 2e-5 5.0 16 qqp_bert_medium_MPO_dynamic2 128 2.8e-6 nompo 10000 10000 10000 prajjwal1/bert-medium Noload --do_train\ --mode=ChildTuning-D\ --topN=6\ --split_num=3
# run_task 1 STS-B 500 2e-5 10.0 16 stsb_bert_medium_MPO_dynamic2 128 2.8e-6 nompo 10000 10000 10000 prajjwal1/bert-medium Noload --do_train\ --mode=ChildTuning-D\ --topN=6\ --split_num=3
# run_task 3 MRPC 500 2e-5 10.0 16 mrpc_bert_medium_MPO_dynamic2 128 2.8e-6 nompo 10000 10000 10000 prajjwal1/bert-medium Noload --do_train\ --mode=ChildTuning-D\ --topN=6\ --split_num=3


#  BERT-base Analysis Experiments

# run_task 0 CoLA 500 2e-5 10.0 16 cola_bert_base_OPF_top1 128 2.8e-6 nompo 10000 10000 10000 bert-base-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=1
# run_task 0 CoLA 500 2e-5 10.0 16 cola_bert_base_OPF_top2 128 2.8e-6 nompo 10000 10000 10000 bert-base-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=2
# run_task 0 CoLA 500 2e-5 10.0 16 cola_bert_base_OPF_top3 128 2.8e-6 nompo 10000 10000 10000 bert-base-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=3
# run_task 3 CoLA 500 2e-5 10.0 16 cola_bert_base_OPF_top4 128 2.8e-6 nompo 10000 10000 10000 bert-base-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=4
# run_task 1 CoLA 500 2e-5 10.0 16 cola_bert_base_OPF_top5 128 2.8e-6 nompo 10000 10000 10000 bert-base-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=5
# run_task 1 CoLA 500 2e-5 10.0 16 cola_bert_base_OPF_top6 128 2.8e-6 nompo 10000 10000 10000 bert-base-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=6
# run_task 0 CoLA 500 2e-5 10.0 16 cola_bert_base_OPF_top7 128 2.8e-6 nompo 10000 10000 10000 bert-base-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=7
# run_task 3 CoLA 500 2e-5 10.0 16 cola_bert_base_OPF_top8 128 2.8e-6 nompo 10000 10000 10000 bert-base-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=8
# run_task 2 CoLA 500 2e-5 10.0 16 cola_bert_base_OPF_top9 128 2.8e-6 nompo 10000 10000 10000 bert-base-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=9
# run_task 4 CoLA 500 2e-5 10.0 16 cola_bert_base_OPF_top10 128 2.8e-6 nompo 10000 10000 10000 bert-base-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=10
# run_task 2 CoLA 500 2e-5 10.0 16 cola_bert_base_OPF_top11 128 2.8e-6 nompo 10000 10000 10000 bert-base-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=11
# run_task 3 CoLA 500 2e-5 10.0 16 cola_bert_base_OPF_top12 128 2.8e-6 nompo 10000 10000 10000 bert-base-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=12


# run_task 4 RTE 500 2e-5 10.0 16 rte_bert_base_OPF_top1 128 2.8e-6 nompo 10000 10000 10000 bert-base-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=1
# run_task 4 RTE 500 2e-5 10.0 16 rte_bert_base_OPF_top2 128 2.8e-6 nompo 10000 10000 10000 bert-base-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=2
# run_task 4 RTE 500 2e-5 10.0 16 rte_bert_base_OPF_top3 128 2.8e-6 nompo 10000 10000 10000 bert-base-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=3
# run_task 5 RTE 500 2e-5 10.0 16 rte_bert_base_OPF_top4 128 2.8e-6 nompo 10000 10000 10000 bert-base-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=4
# run_task 0 RTE 500 2e-5 10.0 16 rte_bert_base_OPF_top5 128 2.8e-6 nompo 10000 10000 10000 bert-base-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=5
# run_task 5 RTE 500 2e-5 10.0 16 rte_bert_base_OPF_top6 128 2.8e-6 nompo 10000 10000 10000 bert-base-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=6
# run_task 5 RTE 500 2e-5 10.0 16 rte_bert_base_OPF_top7 128 2.8e-6 nompo 10000 10000 10000 bert-base-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=7
# run_task 4 RTE 500 2e-5 10.0 16 rte_bert_base_OPF_top8 128 2.8e-6 nompo 10000 10000 10000 bert-base-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=8
# run_task 8 RTE 500 2e-5 10.0 16 rte_bert_base_OPF_top9 128 2.8e-6 nompo 10000 10000 10000 bert-base-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=9
# run_task 1 RTE 500 2e-5 10.0 16 rte_bert_base_OPF_top10 128 2.8e-6 nompo 10000 10000 10000 bert-base-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=10
# run_task 8 RTE 500 2e-5 10.0 16 rte_bert_base_OPF_top11 128 2.8e-6 nompo 10000 10000 10000 bert-base-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=11
# run_task 2 RTE 500 2e-5 10.0 16 rte_bert_base_OPF_top12 128 2.8e-6 nompo 10000 10000 10000 bert-base-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=12




run_task 1 RTE 500 2e-5 10.0 16 rte_bert_base_OPF_top8_split1 128 2.8e-6 nompo 10000 10000 10000 bert-base-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=8\ --split_num=1
run_task 2 RTE 500 2e-5 10.0 16 rte_bert_base_OPF_top8_split2 128 2.8e-6 nompo 10000 10000 10000 bert-base-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=8\ --split_num=2
run_task 4 RTE 500 2e-5 10.0 16 rte_bert_base_OPF_top8_split4 128 2.8e-6 nompo 10000 10000 10000 bert-base-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=8\ --split_num=4
run_task 5 RTE 500 2e-5 10.0 16 rte_bert_base_OPF_top8_split8 128 2.8e-6 nompo 10000 10000 10000 bert-base-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=8\ --split_num=8
# run_task 8 RTE 500 2e-5 10.0 16 rte_bert_base_OPF_top12_split6 128 2.8e-6 nompo 10000 10000 10000 bert-base-uncased Noload --do_train\ --mode=ChildTuning-D\ --topN=12\ --split_num=6
