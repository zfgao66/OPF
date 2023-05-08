#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/8/2 2:49
# @Author  : zfgao
# @File    : run_glue_v1.py
'''
this is for BERT
'''
import dataclasses
import logging
import os
import sys
import torch
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional
sys.path.append('/home/zfgao/Fine-grained-decomposition/')

import numpy as np

from compress_tools_v2.Matrix2MPO_beta import MPO
from transformers import AutoConfig, AutoTokenizer, EvalPrediction, GlueDataset, BartConfig, BartTokenizer
from compress_tools_v2.compress_config import config_dict
# from compress_tools_v2.configuration_bert import BertConfig
from transformers import GlueDataTrainingArguments as DataTrainingArguments
############ transformers V4.0.0
from transformers import BartForSequenceClassification
from trainer_bart import Trainer
# from transformers import Trainer

############ transformers V3.0.2
# from compress_tools_v2.BERTCompress_v6 import BertForSequenceClassificationmy
# from transformers import BertForSequenceClassification
# from trainer import Trainer

from transformers import (
    HfArgumentParser,
#    Trainer,
    TrainingArguments,
    glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
)
from transformers.trainer_callback import (
    TrainerCallback,
    TrainerState,
    TrainerControl
)
logger = logging.getLogger(__name__)
# os.environ["WANDB_DISABLED"] = "true"
# from ChildTuningD import ChildTuningDtrainer

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

def init_config(conf):
    '''
    init value in config.
    '''

    conf.compress = False
    conf.compress_version = 1
    conf.teacher_name_or_path = ""
    conf.alpha = 0.5
    conf.loss_type = ""
    conf.temperature = 0.5
    conf.teacher = False
    conf.row = 0
    conf.loss_we = ""
    conf.mpo=False
    conf.custom_config = None
    return conf

CONVERGE = 10
NO_CONVERGE = 1000
START_SAVE = 384
STOP_RANK = 150
SAVE_STEP = 10
WARM_STEP = 10
class RankHander(TrainerCallback):
    def __init__(self):
        self.window_size = 10
        self.thresh = 0.05
        self.linear_init = 384
        ## stop linear
        self.linear_stop = STOP_RANK

    def on_train_begin(self, args, state, control, **kwargs):
        self.metric_window = [0 for i in range(self.window_size)]
        # self.best_perf = 0.1 # 从当前收敛的满秩中找到(SST-2)
        # self.best_perf = 0.01 # 从当前收敛的满秩中找到(SST-2)
        # self.best_perf = 0.26 # qnli
        # self.best_perf = 0.31 # mnli
        self.best_perf = 0.1 # mrpc
        self.no_converge = 0
        self.converge = 0

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        _ = self.metric_window.pop(0)
        self.metric_window.append(state.loss)
        self.monitor_loss = sum(self.metric_window) / self.window_size
        if self.converge > CONVERGE or self.no_converge > NO_CONVERGE:
            logger.info("Check reset converge and no_converge")
            self.no_converge = 0
            self.converge = 0
        logger.info("Check monitor_loss: {}".format(self.monitor_loss))
        if state.global_step < WARM_STEP:
            logging.info("Check warm step: {}".format(state.global_step))
            return control
        if self.monitor_loss < (self.best_perf + self.thresh) and self.converge < CONVERGE:
            self.converge += 1
            logger.info("Check converge: {}".format(self.converge))
        elif self.monitor_loss > (self.best_perf + self.thresh) and self.no_converge < NO_CONVERGE: # 没有收敛
            self.no_converge += 1
            logger.info("Check no converge {}".format(self.no_converge))
        else:
            if self.converge > CONVERGE-1:
                logger.info("finish one rank attempt, continue ranks")
                self.converge += 1
                # if args.linear_step < START_SAVE and args.linear_step % SAVE_STEP == 0:
                if args.linear_step % SAVE_STEP == 0 or args.attention_step % SAVE_STEP == 0 or args.emb_step % SAVE_STEP == 0:
                    logger.info("Check save at {}...".format(args.linear_step))
                    control.should_save = True
            elif self.no_converge > NO_CONVERGE-1:
                logger.info("finish one rank attempt, failed and stop training...")
                control.should_save = True
                # control.pop_structure = True
                control.should_training_stop = True # 如果用采样来做这个问题就不用pop structure了
                self.no_converge += 1
                # if args.linear_step < self.linear_stop: # 低于最低的rank则停止训练，因为后面基本都不会收敛了
                #     control.should_training_stop = True

                # Initialize our Trainer
            control.change_rank = True

        return control 
@dataclass
class CustomArguments:
    gpu_num : str = field(default='3',
                          metadata = {"help" : "which gpu to use"})
    use_dropout: bool = field(default=False,
                              metadata={"help": "whether use dropout in mpo"})
    use_layernorm: bool = field(default=False,
                                metadata={"help": "whether use layernorm in mpo"})
    use_mpo: bool = field(default=False,
                          metadata={"help": "whether use mpo compress, default is true"})
    warmup_ratio: float = field(default=None,
                                metadata={"help": "ratio of total steps to warmup"})
    mpo_lr: float = field(default=None,
                          metadata={"help": "mpo layer learning rate"})
    word_embed: bool = field(default=False,
                             metadata={"help": "whether use word_embed mpo"})
    embed_size: int = field(default=30522,
                            metadata={"help": "init word embeddings rows"})
    mpo_layers : str = field(default='nompo',
                             metadata={"help" : "layers need to use mpo format"})
    emb_trunc : int = field(default=10000,
                            metadata={"help" : "truncate numbert of embedding"})
    linear_trunc: int = field(default=10000,
                              metadata={"help": "Truncate Rank of linear"})
    attention_trunc : int = field(default=10000,
                                  metadata={"help":"Truncate Rank of attention"})
    teacher : bool = field(default=False,
                           metadata={"help":"Useless"})
    load_layer : str = field(default='',
                             metadata={"help":"Layers which use to load"})
    update_mpo_layer : str = field(default='',
                               metadata={"help":"Layers which to update"})
    fix_all : bool = field(default=False,
                           metadata={"help":"Whether fix all layers except for mpo layers"})
    final_lr : bool = field(default=0,
                            metadata={"help" : "Final lr"})
    tensor_learn : Optional[bool] = field(default=False,
                                metadata={"help":"Whether use tensor learn"})
    rank_step : bool = field(default=False,
                            metadata={"help" : "Whether use rank_step"})
    no_update : str = field(default="",
                            metadata={"help" : "Which layer to be fixed"})
    pooler_trunc : int = field(default=10000)
    step_train : bool = field(default=True)
    load_full : bool = field(default=False)
    #####
    mode : str = field(default=None)
    topN : int = field(default=5)
    weight_types : str = field(default="FFN_1,FFN_2,attention")
    split_num : int = field(default=1)
    dynamic_decom : bool = field(default=False)
def get_parameter_number(net):
    '''
    :param net: model class
    :return: params statistics
    '''
    total_num = sum(p.numel() for p in net.parameters())/1000/1000
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)/1000/1000
    return {'Total(M)': total_num, 'Trainable(M)': trainable_num}

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, CustomArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, custom_args= parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = custom_args.gpu_num
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    try:
        num_labels = glue_tasks_num_labels[data_args.task_name]
        output_mode = glue_output_modes[data_args.task_name]
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    config = BartConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    custom_config = custom_args.__dict__
    for k, v in custom_config.items():
        setattr(config, k, v)
    config.batch_size = training_args.per_device_train_batch_size
    config.max_seq_length = data_args.max_seq_length
    tokenizer = BartTokenizer.from_pretrained(
        model_args.model_name_or_path, #model_args.tokenizer_name if model_args.tokenizer_name else 
        cache_dir=model_args.cache_dir,
    )
    model = BartForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=False,
        config=config,
        cache_dir=None
    )
       
    # if config.mpo_layers:
    #     if 'FFN_1' in config.mpo_layers:
    #         if 'FFN_1' not in config.load_layer:
    #             for i in range(12):
    #                 model.bert.encoder.layer[i].intermediate.from_pretrained_mpo()
    #         for i in range(12):
    #             del model.bert.encoder.layer[i].intermediate.dense
            
    #     if 'FFN_2' in config.mpo_layers:
    #         if 'FFN_2' not in config.load_layer:
    #             for i in range(12):
    #                 model.bert.encoder.layer[i].output.from_pretrained_mpo()
    #         for i in range(12):
    #             del model.bert.encoder.layer[i].output.dense
        
    #     if 'word_embed' in config.mpo_layers:
    #         if 'word_embed' not in config.load_layer:
    #             model.bert.embeddings.from_pretrained_mpo()
    #         else:
    #             logger.info("Check load layer word_embed without from_pretrained...")
    #         del model.bert.embeddings.word_embeddings

    #     if 'attention' in config.mpo_layers:
    #         if 'attention' not in config.load_layer:
    #             for i in range(12):
    #                 model.bert.encoder.layer[i].attention.self.from_pretrained_mpo()
    #                 model.bert.encoder.layer[i].attention.output.from_pretrained_mpo()
    #         for i in range(12):   
    #             del model.bert.encoder.layer[i].attention.self.query
    #             del model.bert.encoder.layer[i].attention.self.key
    #             del model.bert.encoder.layer[i].attention.self.value
    #             del model.bert.encoder.layer[i].attention.output.dense

    #     if 'pooler' in config.mpo_layers:
    #         if 'pooler' not in config.load_layer:
    #             model.bert.pooler.from_pretrained_mpo()
    #         del model.bert.pooler.dense
    # no_update = ["encoder.layer.0","encoder.layer.1.","encoder.layer.2","encoder.layer.3","encoder.layer.4","encoder.layer.5",
                        # "encoder.layer.6","encoder.layer.7","encoder.layer.8","encoder.layer.9","encoder.layer.10","encoder.layer.11"]
    # no_update = ["encoder.layer.{}.".format(i) for i in config.no_update.split(',')]
    # for k,v in model.named_parameters():
    #     for nd in no_update:
    #         if nd in k:
    #             v.requires_grad=False #固定参数
    # logger.info("Check fixed lyaers: {}".format(str(no_update)))
    logger.info("Total Parameter Count: {}M".format(model.num_parameters()/1000/1000))
    logger.info("Total and trainable params: {}".format(str(get_parameter_number(model))))

    # Get datasets
    train_dataset = (
        GlueDataset(data_args, tokenizer=tokenizer, cache_dir=model_args.cache_dir) if training_args.do_train else None
    )
    eval_dataset = (
        GlueDataset(data_args, tokenizer=tokenizer, mode="dev", cache_dir=model_args.cache_dir)
        if training_args.do_eval
        else None
    )
    test_dataset = (
        GlueDataset(data_args, tokenizer=tokenizer, mode="test", cache_dir=model_args.cache_dir)
        if training_args.do_predict
        else None
    )

    def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            if output_mode == "classification":
                preds = np.argmax(p.predictions, axis=1)
            elif output_mode == "regression":
                preds = np.squeeze(p.predictions)
            return glue_compute_metrics(task_name, preds, p.label_ids)

        return compute_metrics_fn

    if custom_args.mpo_lr:
        training_args.mpo_lr = custom_args.mpo_lr
    else:
        training_args.mpo_lr = training_args.learning_rate
    training_args.update_mpo_layer = custom_args.update_mpo_layer
    training_args.fix_all = custom_args.fix_all
    training_args.final_lr = custom_args.final_lr
    training_args.step_train = False
    training_args.step_dir = "/mnt/checkpoint/bert/step_dir"
    training_args.linear_step = config.linear_trunc
    training_args.attention_step = config.attention_trunc
    training_args.emb_step = config.emb_trunc
    ## params for topN
    training_args.topN = config.topN
    training_args.weight_types = config.weight_types
    training_args.mpo_layers = config.mpo_layers
    training_args.load_layer = config.load_layer
    training_args.split_num = config.split_num
    training_args.dynamic_decom = config.dynamic_decom
    training_args.task_name = data_args.task_name
    # Initialize our Trainer
    print("Check mpo_lr= ",training_args.mpo_lr)
    assert config.mode in ['ChildTuning-D', 'fine_grained-D', None]
    logger.info("Childtuning mode: {}".format(config.mode))
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics_fn(data_args.task_name),
        mode=config.mode
    )
    # Training
    if training_args.do_train:
        trainer.train(model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
            eval_datasets.append(
                GlueDataset(mnli_mm_data_args, tokenizer=tokenizer, mode="dev", cache_dir=model_args.cache_dir)
            )

        for eval_dataset in eval_datasets:
            #trainer.compute_metrics = build_compute_metrics_fn(eval_dataset.args.task_name)
            eval_result = trainer.evaluate(eval_dataset=eval_dataset)

            output_eval_file = os.path.join(
                training_args.output_dir, f"eval_results_{eval_dataset.args.task_name}.txt"
            )
            if trainer.is_world_process_zero():
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Eval results {} *****".format(eval_dataset.args.task_name))
                    for key, value in eval_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))

            eval_results.update(eval_result)

    if training_args.do_predict:
        logging.info("*** Test ***")
        test_datasets = [test_dataset]
        if data_args.task_name == "mnli":
            mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
            test_datasets.append(
                GlueDataset(mnli_mm_data_args, tokenizer=tokenizer, mode="test", cache_dir=model_args.cache_dir)
            )

        for test_dataset in test_datasets:
            predictions = trainer.predict(test_dataset=test_dataset).predictions
            if output_mode == "classification":
                predictions = np.argmax(predictions, axis=1)

            output_test_file = os.path.join(
                training_args.output_dir, f"test_results_{test_dataset.args.task_name}.txt"
            )
            if trainer.is_world_process_zero():
                with open(output_test_file, "w") as writer:
                    logger.info("***** Test results {} *****".format(test_dataset.args.task_name))
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if output_mode == "regression":
                            writer.write("%d\t%3.3f\n" % (index, item))
                        else:
                            item = test_dataset.get_labels()[item]
                            writer.write("%d\t%s\n" % (index, item))
    return eval_results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()