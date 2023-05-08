# This scripts trains T5 in a single-task setting.

# We train the model on each single task from the GLUE benchmark by setting the `tasks` and `eval_tasks` 
# to one of GLUE_TASKS=["rte", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "cola"], and report the 
# average obtained test scores.
# export CUDA_VISIBLE_DEVICES=4
# python3 -m torch.distributed.launch --nproc_per_node=4  ./finetune_t5_trainer.py configs/finetune_single_task.json 
function get_gpu_count() {
  str=$1
  array=(${str//,/})
  echo ${#array}
}
function get_init_method() {
  ipaddr=`ifconfig -a|grep inet|grep -v 127.0.0.1|grep -v inet6|awk '{print $2}'|tr -d "addr:"`
  echo ${ipaddr}
}
function run_task_ddp() {
    export CUDA_VISIBLE_DEVICES=$1
    COMMON_ARGS="--output_sub_dir=$3 --mpo_layers=$4 --load_layer=$5 --emb_trunc=$6 --linear_trunc=$7 --attention_trunc=$8 $9"
    nohup python -m torch.distributed.launch --nproc_per_node=`get_gpu_count ${1}` --master_addr=`get_init_method` --master_port=$2 finetune_bert_trainer.py configs/finetune_single_task_bert.json \
    ${COMMON_ARGS}> log_bert/$3_$(date "+%Y%m%d-%H%M%S").log 2>&1 &
}
function run_task() {
    export CUDA_VISIBLE_DEVICES=$1
    export address=$2
    COMMON_ARGS="--output_sub_dir=$3 --mpo_layers=$4 --load_layer=$5 --emb_trunc=$6 --linear_trunc=$7 --attention_trunc=$8 $9"
    nohup python -u finetune_bert_trainer.py configs/finetune_single_task_bert.json \
    ${COMMON_ARGS}> log_bert/$3_$(date "+%Y%m%d-%H%M%S").log 2>&1 &
}

run_task 5 12347 bert_sinlge_cola word_embed,FFN_1,FFN_2,attention noload 30000 30000 30000