set -x
base_dir=/home/zfgao/work/Fine-grained-decomposition/bart
data_dir_base=/mnt/zfgao/data/nlp_data/GLUE
check_point_dir=/mnt/zfgao/checkpoint/sshlab/bart

export WANDB_PROJECT=BART-base-MPO
export OMP_NUM_THREADS=16
echo $gpu_num
function run_task() {
  export CUDA_VISIBLE_DEVICES=$1
  COMMON_ARGS="--data_dir="$data_dir_base/$2" --model_name_or_path=${14} --tokenizer_name=bert-base-uncased --evaluation_strategy=steps --eval_steps=100 --logging_steps=50 --overwrite_output_dir --save_steps=50000 --gpu_num=$1 --task_name=$2 --warmup_step=$3 --run_name=$7 --learning_rate=$4 --num_train_epochs=$5 --per_device_train_batch_size=$6 --output_dir="$check_point_dir/$7" --max_seq_length=$8 --mpo_lr=$9 --mpo_layers=${10} --emb_trunc=${11} --linear_trunc=${12} --attention_trunc=${13} --load_layer=${15} --update_mpo_layer=${16} ${17}"
  nohup python $base_dir/run_glue_bart.py \
      --do_eval ${COMMON_ARGS} > $base_dir/log/bart_base/$7.log 2>&1 &
}

# run_task 2 SST-2 500 2e-5 3.0 32 sst_baseline 128 2.8e-6 nompo 1000 1000 1000 facebook/bart-base Noload noupdate --do_train\ --metric_for_best_model="acc"\ --eval_steps=500
# run_task 8 QQP 500 2e-5 3.0 32 qqp_baseline 128 2.8e-6 nompo 1000 1000 1000 facebook/bart-base Noload noupdate --do_train\ --per_device_eval_batch_size=2\ --metric_for_best_model="acc"\ --eval_steps=50
# run_task 8 QQP 500 2e-5 3.0 32 qqp_baseline 128 2.8e-6 nompo 1000 1000 1000 facebook/bart-base Noload noupdate --do_train\ --per_device_eval_batch_size=2\ --metric_for_best_model="acc"\ --eval_steps=50
# run_task 1 QQP 6796 1e-5 3.0 32 qqp_baseline_2 128 1e-5 nompo 1000 1000 1000 facebook/bart-base Noload noupdate --do_train\ --do_eval\ --per_device_eval_batch_size=16\ --metric_for_best_model="acc"\ --eval_steps=20000\ --max_steps=113272
# run_task 1 QQP 6796 1e-5 3.0 32 qqp_baseline_predict 128 1e-5 nompo 1000 1000 1000 ModelTC/bart-base-qqp Noload noupdate --do_predict\ --per_device_eval_batch_size=16\ --metric_for_best_model="acc"\ --eval_steps=20000\ --max_steps=113272

# BART-Base Dynamic
# run_task 0 MRPC 500 2e-5 5.0 32 mrpc_bart_base_top8_splitnum2 128 2.8e-6 nompo 1000 1000 1000 facebook/bart-base Noload noupdate --do_train\ --metric_for_best_model="f1"\ --eval_steps=500\ --mode=ChildTuning-D\ --dynamic_decom\ --topN=8\ --split_num=2
# run_task 1 CoLA 500 2e-5 5.0 32 cola_bart_base_top8_splitnum2 128 2.8e-6 nompo 1000 1000 1000 facebook/bart-base Noload noupdate --do_train\ --metric_for_best_model="mcc"\ --eval_steps=500\ --mode=ChildTuning-D\ --dynamic_decom\ --topN=8\ --split_num=2
# run_task 2 SST-2 500 2e-5 5.0 32 sst2_bart_base_top8_splitnum2 128 2.8e-6 nompo 1000 1000 1000 facebook/bart-base Noload noupdate --do_train\ --metric_for_best_model="acc"\ --eval_steps=500\ --mode=ChildTuning-D\ --dynamic_decom\ --topN=8\ --split_num=2
# run_task 3 MNLI 500 2e-5 5.0 32 mnli_bart_base_top8_splitnum2 128 2.8e-6 nompo 1000 1000 1000 facebook/bart-base Noload noupdate --do_train\ --metric_for_best_model="acc"\ --eval_steps=500\ --mode=ChildTuning-D\ --dynamic_decom\ --topN=8\ --split_num=2
# run_task 7 QNLI 500 2e-5 5.0 32 qnli_bart_base_top8_splitnum2 128 2.8e-6 nompo 1000 1000 1000 facebook/bart-base Noload noupdate --do_train\ --metric_for_best_model="acc"\ --eval_steps=500\ --mode=ChildTuning-D\ --dynamic_decom\ --topN=8\ --split_num=2
# run_task 5 QQP 500 2e-5 5.0 32 qqp_bart_base_top8_splitnum2 128 2.8e-6 nompo 1000 1000 1000 facebook/bart-base Noload noupdate --do_train\ --metric_for_best_model="acc"\ --eval_steps=500\ --mode=ChildTuning-D\ --dynamic_decom\ --topN=8\ --split_num=2
# run_task 8 RTE 500 2e-5 5.0 32 rte_bart_base_top8_splitnum2 128 2.8e-6 nompo 1000 1000 1000 facebook/bart-base Noload noupdate --do_train\ --metric_for_best_model="acc"\ --eval_steps=500\ --mode=ChildTuning-D\ --dynamic_decom\ --topN=8\ --split_num=2

run_task 7 STS-B 500 2e-5 5.0 32 stsb_bart_base_top8_splitnum2 128 2.8e-6 nompo 1000 1000 1000 facebook/bart-base Noload noupdate --do_train\ --metric_for_best_model="spearman"\ --eval_steps=500\ --mode=ChildTuning-D\ --dynamic_decom\ --topN=8\ --split_num=2

