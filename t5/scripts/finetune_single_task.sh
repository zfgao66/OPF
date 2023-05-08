# This scripts trains T5 in a single-task setting.

# We train the model on each single task from the GLUE benchmark by setting the `tasks` and `eval_tasks` 
# to one of GLUE_TASKS=["rte", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "cola"], and report the 
# average obtained test scores.
# export CUDA_VISIBLE_DEVICES=4
# python3 -m torch.distributed.launch --nproc_per_node=4  ./finetune_t5_trainer.py configs/finetune_single_task.json 
export WANDB_PROJECT=fermi_t5_hyperformer
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
    nohup python -m torch.distributed.launch --nproc_per_node=`get_gpu_count ${1}` --master_addr=`get_init_method` --master_port=$2 finetune_t5_trainer.py configs/finetune_single_task.json \
    ${COMMON_ARGS}> log/$3_$(date "+%Y%m%d-%H%M%S").log 2>&1 &
}
function run_task() {
    export CUDA_VISIBLE_DEVICES=$1
    export address=$2
    COMMON_ARGS="--output_sub_dir=$3 --mpo_layers=$4 --load_layer=$5 --emb_trunc=$6 --linear_trunc=$7 --attention_trunc=$8 $9"
    nohup python finetune_t5_trainer.py configs/finetune_single_task.json \
    ${COMMON_ARGS}> log/$3_$(date "+%Y%m%d-%H%M%S").log 2>&1 &
}

function run_task_large() {
    export CUDA_VISIBLE_DEVICES=$1
    export address=$2
    COMMON_ARGS="--output_sub_dir=$3 --mpo_layers=$4 --load_layer=$5 --emb_trunc=$6 --linear_trunc=$7 --attention_trunc=$8 $9"
    nohup python finetune_t5_trainer.py configs/finetune_single_task_large.json \
    ${COMMON_ARGS}> log/$3_$(date "+%Y%m%d-%H%M%S").log 2>&1 &
}


# V-100 For T5-Large
# run_task_large 8 123521 t5_large_switch_4exp_mnli mlp noload 30000 30000 30000 --tasks=mnli\ --eval_tasks=mnli\ --output_dir=/mnt/zfgao/checkpoint/mnli\ --moe_type=switch\ --num_experts=3

# 3090 For T5-Base
# run_task 0 135967 t5_base_moe_8exp_mnli nompo noload 30000 30000 30000 --tasks=mnli\ --eval_tasks=mnli\ --output_dir=/mnt/zfgao/checkpoint/MoE/mnli\ --moe_type=moe\ --num_experts=8

# run_task 1 135977 t5_base_moe_8exp_qnli nompo noload 30000 30000 30000 --tasks=qnli\ --eval_tasks=qnli\ --output_dir=/mnt/zfgao/checkpoint/MoE/qnli\ --moe_type=moe\ --num_experts=8

# run_task 2 135978 t5_base_moe_8exp_sst2 nompo noload 30000 30000 30000 --tasks=sst2\ --eval_tasks=sst2\ --output_dir=/mnt/zfgao/checkpoint/MoE/sst2\ --moe_type=moe\ --num_experts=8

# run_task 3 135979 t5_base_moe_8exp_rte nompo noload 30000 30000 30000 --tasks=rte\ --eval_tasks=rte\ --output_dir=/mnt/zfgao/checkpoint/MoE/rte\ --moe_type=moe\ --num_experts=8

# run_task 4 135929 t5_base_moe_8exp_qqp nompo noload 30000 30000 30000 --tasks=qqp\ --eval_tasks=qqp\ --output_dir=/mnt/zfgao/checkpoint/MoE/qqp\ --moe_type=moe\ --num_experts=8

# run_task 6 165929 t5_base_moe_8exp_cola nompo noload 30000 30000 30000 --tasks=cola\ --eval_tasks=cola\ --output_dir=/mnt/zfgao/checkpoint/MoE/cola\ --moe_type=moe\ --num_experts=8

# run_task 7 165939 t5_base_moe_8exp_mrpc nompo noload 30000 30000 30000 --tasks=mrpc\ --eval_tasks=mrpc\ --output_dir=/mnt/zfgao/checkpoint/MoE/mrpc\ --moe_type=moe\ --num_experts=8

# run_task 8 165949 t5_base_moe_8exp_stsb nompo noload 30000 30000 30000 --tasks=stsb\ --eval_tasks=stsb\ --output_dir=/mnt/zfgao/checkpoint/MoE/stsb\ --moe_type=moe\ --num_experts=8

# run_task 8 162949 t5_base_moe_8exp_wnli nompo noload 30000 30000 30000 --tasks=wnli\ --eval_tasks=wnli\ --output_dir=/mnt/zfgao/checkpoint/MoE/wnli\ --moe_type=moe\ --num_experts=8

##################################################################################################################################
# MPOE version

# 3090 For T5-Base
# run_task 0 135967 t5_base_MPOE++_8exp_mnli mlp noload 30000 30000 30000 --tasks=mnli\ --eval_tasks=mnli\ --output_dir=/mnt/zfgao/checkpoint/MPOE/mnli\ --moe_type=switch\ --num_experts=8

# run_task 1 135977 t5_base_MPOE++_16exp_qnli mlp noload 30000 30000 30000 --tasks=qnli\ --eval_tasks=qnli\ --output_dir=/mnt/zfgao/checkpoint/MPOE/qnli\ --moe_type=switch\ --num_experts=16

# run_task 2 135978 t5_base_MPOE++_16exp_sst2 mlp noload 30000 30000 30000 --tasks=sst2\ --eval_tasks=sst2\ --output_dir=/mnt/zfgao/checkpoint/MPOE/sst21\ --moe_type=switch\ --num_experts=16

# run_task 3 135979 t5_base_MPOE++_12exp_rte mlp noload 30000 30000 30000 --tasks=rte\ --eval_tasks=rte\ --output_dir=/mnt/zfgao/checkpoint/MPOE/rte33\ --moe_type=switch\ --num_experts=12

# run_task 4 135929 t5_base_MPOE++_16exp_mnli mlp noload 30000 30000 30000 --tasks=mnli\ --eval_tasks=mnli\ --output_dir=/mnt/zfgao/checkpoint/MPOE/mnli4\ --moe_type=switch\ --num_experts=16

# run_task 5 165929 t5_base_MPOE++_16exp_cola mlp noload 30000 30000 30000 --tasks=cola\ --eval_tasks=cola\ --output_dir=/mnt/zfgao/checkpoint/MPOE/cola1\ --moe_type=switch\ --num_experts=16

# run_task 6 165939 t5_base_MPOE++_16exp_mrpc mlp noload 30000 30000 30000 --tasks=mrpc\ --eval_tasks=mrpc\ --output_dir=/mnt/zfgao/checkpoint/MPOE/mrpc1\ --moe_type=switch\ --num_experts=16

# run_task 6 165949 t5_base_MPOE++_16exp_stsb mlp noload 30000 30000 30000 --tasks=stsb\ --eval_tasks=stsb\ --output_dir=/mnt/zfgao/checkpoint/MPOE/stsb4\ --moe_type=switch\ --num_experts=16

# run_task 6 162949 t5_base_MPOE++_12exp_mnli_moe mlp noload 30000 30000 30000 --tasks=mnli\ --eval_tasks=mnli\ --output_dir=/mnt/zfgao/checkpoint/MPOE/mnli_moe\ --moe_type=moe\ --num_experts=12


# run_task 3 135972 t5_base_MPOE++_16exp_mnli mlp noload 30000 30000 30000 --tasks=mnli\ --eval_tasks=mnli\ --output_dir=/mnt/zfgao/checkpoint/MPOE/mnli\ --moe_type=switch\ --num_experts=16


# run_task 0 165929 t5_base_moe_8exp_cola_test nompo noload 30000 30000 30000 --tasks=cola\ --eval_tasks=cola\ --output_dir=/mnt/zfgao/checkpoint/MoE/cola_test\ --moe_type=moe\ --num_experts=8\ --load_experiment=/mnt/zfgao/checkpoint/MoE/cola

# run_task 7 135167 t5_base_switch_8exp_cola_test nompo noload 30000 30000 30000 --tasks=cola\ --eval_tasks=cola\ --output_dir=/mnt/zfgao/checkpoint/cola\ --moe_type=switch\ --num_experts=8\ --load_experiment=/mnt/zfgao/checkpoint/cola_shannon


# run_task 1 135123 t5_base_switch_8exp_cola_test_time mlp noload 30000 30000 30000 --tasks=cola\ --eval_tasks=cola\ --output_dir=/mnt/zfgao/checkpoint/cola_20220717\ --moe_type=switch\ --num_experts=8
# run_task 0 165929 t5_base_MPOE++_16exp_cola_test mlp noload 30000 30000 30000 --tasks=cola\ --eval_tasks=cola\ --output_dir=/mnt/zfgao/checkpoint/MPOE/cola1\ --moe_type=switch\ --num_experts=16\ --load_experiment=/mnt/zfgao/checkpoint/MoE/cola


run_task 0 165929 t5_base_switch_8exp_cola_test_3 nompo noload 30000 30000 30000 --tasks=cola\ --eval_tasks=cola\ --output_dir=/mnt/zfgao/checkpoint/MoE/cola_20220717_2\ --moe_type=switch\ --num_experts=8\ --load_experiment=/mnt/zfgao/checkpoint/MoE/cola_20220717

# run_task 8 135123 t5_base_switch_8exp_cola_test_mpo_3 mlp noload 30000 30000 30000 --tasks=cola\ --eval_tasks=cola\ --output_dir=/mnt/zfgao/checkpoint/Switch/cola_20220717_3\ --moe_type=switch\ --num_experts=8\ --load_experiment=/mnt/zfgao/checkpoint/Switch/cola_20220717