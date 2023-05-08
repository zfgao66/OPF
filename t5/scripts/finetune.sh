# This script trains multi-task T5 on the GLUE benchmark.
# export CUDA_VISIBLE_DEVICES=0,1
# python3 -m torch.distributed.launch --nproc_per_node=2  ./finetune_t5_trainer.py configs/finetune.json 
# ------- new task
export WANDB_PROJECT=t5_hyperformer
set -x
check_point_dir=/mnt/liupeiyu/checkpoint/hyperformer_exp
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
    nohup python -m torch.distributed.launch --nproc_per_node=`get_gpu_count ${1}` --master_addr=183.174.229.154 --master_port=$2 finetune_t5_trainer.py configs/finetune.json \
    ${COMMON_ARGS}> log/$3_$(date "+%Y%m%d-%H%M%S").log 2>&1 &
}
function run_task() {
    export CUDA_VISIBLE_DEVICES=$1
    COMMON_ARGS="--output_sub_dir=$3 --mpo_layers=$4 --load_layer=$5 --emb_trunc=$6 --linear_trunc=$7 --attention_trunc=$8 $9"
    nohup python -u finetune_t5_trainer.py configs/finetune.json \
    ${COMMON_ARGS}> log/$3_$(date "+%Y%m%d-%H%M%S").log 2>&1 &
}
# run_task_ddp 4,5,6,7 12345 t5_mpo word_embed,mlp,attention noload 30000 30000 30000 --tensor_learn
# run_task_ddp 0,1,2,3,4,5,6,7 12345 t5_mpo2 word_embed,mlp,attention noload 30000 30000 30000 --tensor_learn
# run_task_ddp 0,1,2,3,4,5,6,7 12345 t5_mpo2_test word_embed,mlp,attention word_embed,mlp,attention 30000 30000 30000 --tensor_learn
# run_task_ddp 0,1,2,3,4,5,6,7 12345 t5_mpo3 word_embed,mlp,attention noload 30000 30000 30000 --tensor_learn
# run_task_ddp 0,1,2,3,4,5,6,7 12345 t5_mpo4 word_embed,mlp,attention noload 30000 30000 30000 --tensor_learn
# run_task_ddp 0,1,2,3 12345 t5_mpo6 nompo noload 30000 30000 30000 --tensor_learn
# run_task_ddp 0,1,2,3 12345 t5_mpo7 word_embed,mlp,attention noload 30000 30000 30000 --tensor_learn
# run_task_ddp 0,1,2,3,4,5,6,7 12345 t5_mpo7 word_embed,mlp,attention noload 30000 30000 30000
# run_task_ddp 0,1,2,3,4,5,6,7 12345 t5_mpo8_full word_embed,mlp,attention noload 30000 30000 30000 --tensor_learn
# run_task_ddp 0,1,2,3 12345 t5_mpo10_full word_embed,mlp,attention noload 30000 30000 30000
# run_task_ddp 4,5,6,7 12346 t5_basleine nompo noload 30000 30000 30000
# run_task_ddp 4,5,6,7 12346 t5_mpo_baseline word_embed,mlp,attention noload 30000 30000 30000\ --load_experiment=/mnt/liupeiyu/checkpoint/hyperformer
# run_task_ddp 4,5,6,7 12346 t5_mpo11_full mlp noload 30000 30000 30000
# run_task_ddp 4,5,6,7 12346 t5_mpo12_full mlp noload 30000 30000 30000
# run_task_ddp 5,6,7 12347 t5_mpo13 mlp noload 30000 30000 30000 --moe_type=moe\ --num_experts=4\ --tensor_learn
# run_task_ddp 0,1,2,3 12348 t5_mpo13_lr mlp noload 30000 30000 30000 --moe_type=moe\ --num_experts=4\ --tensor_learn
# run_task_ddp 0,1,2,3,4,5,6,7 12348 t5_mpo13 mlp noload 30000 30000 30000 --moe_type=moe\ --num_experts=4
# run_task_ddp 0,1,2,3 12348 t5_mpo13_lr_full mlp noload 30000 30000 30000 --moe_type=moe\ --num_experts=4
# run_task_ddp 4,5,6,7 12349 t5_moe nompo noload 30000 30000 30000 --moe_type=moe\ --num_experts=4
# run_task_ddp 0,1,2,3 12349 t5_swi nompo noload 30000 30000 30000 --moe_type=switch\ --num_experts=4
# run_task_ddp 4,5,6,7 12350 t5_swi_lr nompo noload 30000 30000 30000 --moe_type=switch\ --num_experts=4
# run_task_ddp 0,1,2,3 12349 t5_swi_mpo_lr mlp noload 30000 30000 30000 --moe_type=switch\ --num_experts=4\ --tensor_learn
# run_task_ddp 4,5,6,7 12351 t5_swi_4exp_1.5e4 nompo noload 30000 30000 30000 --moe_type=switch\ --num_experts=4
# run_task_ddp 4,5,6,7 12351 t5_swi_4exp_mpo_4exp_3e-5 mlp noload 30000 30000 30000 --moe_type=switch\ --num_experts=4
# run_task_ddp 0,1,2,3 12346 t5_basleine_2 nompo noload 30000 30000 30000
# run_task_ddp 4,5,6,7 12347 t5_switch_swidropout0.4 nompo noload 30000 30000 30000 --moe_type=switch\ --num_experts=4\ --switch_dropout=0.4
# run_task_ddp 0,1,2,3 12348 t5_switch_baseline_2 nompo noload 30000 30000 30000 --moe_type=switch\ --num_experts=4\ --is_scale_prob
# run_task_ddp 0,1,2,3 12348 t5_switch_baseline_3 nompo noload 30000 30000 30000 --moe_type=switch\ --num_experts=4
# run_task_ddp 0,1,2,3 12348 t5_switch_baseline_4 nompo noload 30000 30000 30000 --moe_type=switch\ --num_experts=4

# NAACL
# run_task_ddp 6,7 12351 t5_swi_4exp_mpo_4exp_3e-5 mlp noload 30000 30000 30000 --moe_type=switch\ --num_experts=4\ --tensor_learn
# run_task 6 12351 t5_swi_4exp_mpo_4exp_3e-5 mlp noload 30000 30000 30000 --moe_type=switch\ --num_experts=4\ --tensor_learn
# run_task 7 12351 t5_swi_4exp_mpo_4exp_3e-5_test mlp mlp 30000 30000 30000 --moe_type=switch\ --num_experts=4\ --tensor_learn\ --load_experiment=/mnt/liupeiyu/checkpoint/hyperformer_exp/t5_swi_4exp_mpo_4exp_3e-5
# run_task 7 12351 t5_swi_4exp_mpo_4exp_3e-5_test2 mlp noload 30000 30000 30000 --moe_type=switch\ --num_experts=4\ --tensor_learn\ --load_experiment=/mnt/liupeiyu/checkpoint/hyperformer_exp/t5_swi_4exp_mpo_4exp_3e-5
run_task 7 12351 t5_swi_4exp_mpo_4exp_3e-5_test3 mlp mlp 30000 30000 30000 --moe_type=switch\ --num_experts=4\ --tensor_learn\ --load_experiment=/mnt/liupeiyu/checkpoint/hyperformer_exp/t5_swi_4exp_mpo_4exp_3e-5

# run_task 4 12351 t5_swi_4exp_mpo_4exp_3e-5 mlp noload 30000 30000 30000 --moe_type=switch\ --num_experts=4
