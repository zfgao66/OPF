# This scripts trains T5 in a single-task setting.

# We train the model on each single task from the GLUE benchmark by setting the `tasks` and `eval_tasks` 
# to one of GLUE_TASKS=["rte", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "cola"], and report the 
# average obtained test scores.
# export CUDA_VISIBLE_DEVICES=4
# python3 -m torch.distributed.launch --nproc_per_node=4  ./finetune_t5_trainer.py configs/finetune_single_task.json 
export WANDB_PROJECT=T5-Base-FTours
export OMP_NUM_THREADS=20 

function get_gpu_count() {
  str=$1
  array=(${str//,/})
  echo ${#array}
}
function get_init_method() {
  ipaddr=`ifconfig -a|grep inet|grep -v 172.17.0.1|grep -v 127.0.0.1|grep -v inet6|awk '{print $2}'|tr -d "addr:"`
  echo ${ipaddr}
}
function run_task_ddp() {
    export CUDA_VISIBLE_DEVICES=$1
    COMMON_ARGS="--output_sub_dir=$3 --mpo_layers=$4 --load_layer=$5 --emb_trunc=$6 --linear_trunc=$7 --attention_trunc=$8 $9"
    nohup python -m torch.distributed.launch --nproc_per_node=`get_gpu_count ${1}` --master_addr=`get_init_method` --master_port=$2 finetune_t5_trainer.py configs/finetune_single_task.json \
    ${COMMON_ARGS}> log/t5_base_origin/$3_$(date "+%Y%m%d-%H%M%S").log 2>&1 &
}
function run_task() {
    export CUDA_VISIBLE_DEVICES=$1
    export address=$2
    COMMON_ARGS="--output_sub_dir=$3 --mpo_layers=$4 --load_layer=$5 --emb_trunc=$6 --linear_trunc=$7 --attention_trunc=$8 --tasks=$9 --eval_tasks=$9 ${10}"
    nohup python finetune_t5_trainer.py configs/finetune_single_task.json \
    ${COMMON_ARGS}> log/t5_large/$3.log 2>&1 &
}

# function run_task_large() {
#     export CUDA_VISIBLE_DEVICES=$1
#     export address=$2
#     COMMON_ARGS="--output_sub_dir=$3 --mpo_layers=$4 --load_layer=$5 --emb_trunc=$6 --linear_trunc=$7 --attention_trunc=$8 $9"
#     nohup python finetune_t5_trainer.py configs/finetune_single_task.json \
#     ${COMMON_ARGS}> log/$3.log 2>&1 &
# }


# V-100 For T5-Large
# run_task_large 8 123521 t5_large_switch_4exp_mnli mlp noload 30000 30000 30000 --tasks=mnli\ --eval_tasks=mnli\ --output_dir=/mnt/zfgao/checkpoint/mnli\ --moe_type=switch\ --num_experts=3

# For T5-Base
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

# For T5-Base
# run_task 0 135978 t5_base_MPOE++_16exp_sst2 mlp noload 30000 30000 30000 --tasks=sst2\ --eval_tasks=sst2\ --output_dir=/mnt/zfgao/checkpoint/MPOE/sst2\ --moe_type=switch\ --num_experts=16

# run_task 1 135977 t5_base_moe_8exp_qnli mlp noload 30000 30000 30000 --tasks=qnli\ --eval_tasks=qnli\ --output_dir=/mnt/zfgao/checkpoint/MPOE/qnli\ --moe_type=switch\ --num_experts=8

# run_task 2 135978 t5_base_moe_8exp_sst2 mlp noload 30000 30000 30000 --tasks=sst2\ --eval_tasks=sst2\ --output_dir=/mnt/zfgao/checkpoint/MPOE/sst2\ --moe_type=switch\ --num_experts=8

# run_task 3 135979 t5_base_MPOE++_16exp_rte mlp noload 30000 30000 30000 --tasks=rte\ --eval_tasks=rte\ --output_dir=/mnt/zfgao/checkpoint/MPOE/rte\ --moe_type=switch\ --num_experts=16

# run_task 4 135929 t5_base_moe_8exp_qqp mlp noload 30000 30000 30000 --tasks=qqp\ --eval_tasks=qqp\ --output_dir=/mnt/zfgao/checkpoint/MPOE/qqp\ --moe_type=switch\ --num_experts=8

# run_task 6 165929 t5_base_moe_8exp_cola mlp noload 30000 30000 30000 --tasks=cola\ --eval_tasks=cola\ --output_dir=/mnt/zfgao/checkpoint/MPOE/cola\ --moe_type=switch\ --num_experts=8

# run_task 7 165939 t5_base_moe_8exp_mrpc mlp noload 30000 30000 30000 --tasks=mrpc\ --eval_tasks=mrpc\ --output_dir=/mnt/zfgao/checkpoint/MPOE/mrpc\ --moe_type=switch\ --num_experts=8

# run_task 8 165949 t5_base_moe_8exp_stsb mlp noload 30000 30000 30000 --tasks=stsb\ --eval_tasks=stsb\ --output_dir=/mnt/zfgao/checkpoint/MPOE/stsb\ --moe_type=switch\ --num_experts=8

# run_task 0 12347 t5_large_cola word_embed,mlp,attention noload 30000 30000 30000
# run_task 0 12347 t5_large_qnli word_embed,mlp,attention noload 30000 30000 30000
# run_task 0 12347 t5_large_cola word_embed,mlp,attention noload 30000 30000 30000
# run_task 0 12347 t5_large_sst2_origin nompo noload 30000 30000 30000
# run_task 1 12347 t5_large_qqp_origin nompo noload 30000 30000 30000
# run_task 0 123 t5_large_sst2_origin nompo noload 30000 30000 30000
# run_task_ddp 1,2,3 13223 t5_large_wnli_origin nompo noload 30000 30000 30000

# run_task_ddp 0,4 135967 t5_base_origin_rte nompo noload 30000 30000 30000 --tasks=rte\ --eval_tasks=rte
# run_task 0 135967 t5_base_moe_2exp_mrpc nompo noload 30000 30000 30000 --tasks=mrpc\ --eval_tasks=mrpc\ --output_dir=/home/zfgao/checkpoint/mrpc\ --moe_type=moe\ --num_experts=2

# run_task 0 135967 t5_base_switch_8exp_mnli nompo noload 30000 30000 30000 --tasks=mnli\ --eval_tasks=mnli\ --output_dir=/home/zfgao/checkpoint/mnli2\ --moe_type=switch\ --num_experts=8




# ************************************************************************************

# T5-Base-MPO
# run_task 4 135918 cola_mpo_t5_large mlp,attention noload 10000 10000 10000 sst2 --output_dir=/mnt/zfgao/checkpoint/Over-parameter-t5-large/sst21
# run_task 6 135928 mrpc_mpo_t5_large mlp,attention noload 10000 10000 10000 mrpc --output_dir=/mnt/zfgao/checkpoint/Over-parameter-t5-large/mrpc1
# run_task 7 135938 stsb_mpo_t5_large mlp,attention noload 10000 10000 10000 stsb --output_dir=/mnt/zfgao/checkpoint/Over-paramete-t5-large/stsb1
run_task 8 135948 rte_mpo_t5_large2 mlp,attention noload 10000 10000 10000 rte --output_dir=/mnt/zfgao/checkpoint/Over-parameter-t5-large/rte2

# run_task 6 13592 t5_base_sst2_mpo2 mlp,attention noload 30000 30000 30000 --tasks=sst2\ --eval_tasks=sst2\ --output_dir=/mnt/zfgao/checkpoint/Over-parameter/sst22