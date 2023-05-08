import sys
import torch
import datasets
import json
import logging
import os
from pathlib import Path
sys.path.append('/home/zfgao/work/MPOE_T5/')
from transformers import AutoTokenizer, HfArgumentParser, set_seed
from transformers.trainer_utils import EvaluationStrategy
from transformers import glue_tasks_num_labels

from MPOE.third_party.models import AutoConfig, BertForSequenceClassification
from MPOE.third_party.trainers import BERTTrainer
from MPOE.adapters import AdapterController, AutoAdapterConfig
from MPOE.data import AutoTask
from MPOE.third_party.utils import TaskCollator, check_output_dir
from MPOE.metrics import build_compute_metrics_fn
from MPOE.training_args import Seq2SeqTrainingArguments, ModelArguments, DataTrainingArguments, \
    AdapterTrainingArguments, CustomArguments
from MPOE.utils import freezing_params, get_last_checkpoint_path, create_dir,\
    handle_metrics, get_training_args
from mpo_lab.MPOtorch import MPO
logger = logging.getLogger(__name__)


def remove_rank_info_from_argv(args):
    extra_parameters = {}
    if args[1].startswith("--local_rank"):
        extra_parameters.update({'local_rank': int(args[1].split('=')[-1])})
        del args[1]
    return extra_parameters

def main():
    # See all possible arguments in src/transformers/training_args.py or by passing
    # the --help flag to this script. We now keep distinct sets of args, for a cleaner
    # separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments, AdapterTrainingArguments))
    parser_custom = HfArgumentParser((CustomArguments))
    
    # For running on multiple gpus with torch.distributed.launch, it adds a local_rank paramter, to allow the parser
    # still use the config file, we add the local_rank to the config file.
    if len(sys.argv) > 2 and sys.argv[1].startswith("--local_rank") and (sys.argv[2].endswith(".json")):
        rank_info = remove_rank_info_from_argv(sys.argv)
        args_dict = json.loads(Path(sys.argv[1]).read_text())
        args_dict.update(rank_info)
        model_args, data_args, training_args, adapter_args = parser.parse_dict(args_dict)
        custom_args = parser_custom.parse_args(sys.argv[2:])
    elif sys.argv[1].endswith(".json"):
        custom_args = parser_custom.parse_args(sys.argv[2:])
        logger.warning("config path: %s", sys.argv[1])
        args_dict = json.loads(Path(sys.argv[1]).read_text())
        model_args, data_args, training_args, adapter_args = parser.parse_dict(args_dict)
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        logger.warning("config path: %s", sys.argv[1])
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, adapter_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, adapter_args = parser.parse_args_into_dataclasses()
    training_args.output_dir = os.path.join(training_args.output_dir, custom_args.output_sub_dir)
    os.makedirs(training_args.output_dir, exist_ok=True)
    check_output_dir(training_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.setLevel(logging.INFO)
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

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    try:
        num_labels = glue_tasks_num_labels[data_args.tasks[0]]
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.tasks[0]))

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else \
            model_args.model_name_or_path,
            num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    custom_config = custom_args.__dict__
    for k, v in custom_config.items():
        setattr(config, k, v)
    config.batch_size = training_args.per_device_train_batch_size
    config.max_seq_length = data_args.max_source_length
    extra_model_params = ("encoder_layerdrop", "decoder_layerdrop", "dropout",
                          "attention_dropout",  "train_adapters")
    for p in extra_model_params:
        if getattr(training_args, p, None):
            assert hasattr(config, p), f"({config.__class__.__name__}) doesn't have a `{p}` attribute"
            setattr(config, p, getattr(training_args, p))

    # Gets the adapter config and updates the specified parameters.
    if training_args.train_adapters:
        adapter_config = AutoAdapterConfig.get(adapter_args.adapter_config_name)
        adapter_config.input_dim = config.d_model
        adapter_config.tasks = data_args.tasks
        adapter_config.task_to_adapter = {task:adapter for task, adapter in zip(data_args.tasks, data_args.adapters)} if data_args.adapters is not None else None
        # If this is a parametric task embedding this mapping makes sense, but in case we use any task embeddings,
        # then, we do not need any mapping as we use the pretrained task embeddings.
        adapter_config.task_to_embeddings = {task:embedding for task, embedding in zip(data_args.tasks, data_args.task_embeddings)}\
             if (data_args.task_embeddings is not None) else None
        extra_adapter_params = ("task_embedding_dim",
                                "add_layer_norm_before_adapter",
                                "add_layer_norm_after_adapter",
                                "reduction_factor",
                                "hidden_dim",
                                "non_linearity",
                                "train_task_embeddings",
                                "projected_task_embedding_dim",
                                "task_hidden_dim",
                                "conditional_layer_norm",
                                "train_adapters_blocks",
                                "unique_hyper_net",
                                "unique_hyper_net_layer_norm",
                                "efficient_unique_hyper_net")
        for p in extra_adapter_params:
            if hasattr(adapter_args, p) and hasattr(adapter_config, p):
                setattr(adapter_config, p, getattr(adapter_args, p))
            else:
                logger.warning(f"({adapter_config.__class__.__name__}) doesn't have a `{p}` attribute")
        adapter_config.device = training_args.device
    else:
        adapter_config = None

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else \
            model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    if model_args.not_load_t5_checkpoint:
        model = BertForSequenceClassification(config=config, adapter_config=adapter_config)
    else:
        last_checkpoint_path = training_args.output_dir if not config.load_experiment else get_last_checkpoint_path(config.load_experiment)
        model_path = model_args.model_name_or_path if ((training_args.optimize_from_scratch and not training_args.optimize_from_scratch_with_loading_model) or not os.path.exists(os.path.join(last_checkpoint_path, 'pytorch_model.bin')))\
            else last_checkpoint_path
        logger.warning("model path loaded from : %s", model_path)
        model = BertForSequenceClassification.from_pretrained(
            model_path,
            from_tf=".ckpt" in model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
            adapter_config=adapter_config
        )

    ########## mpo setting
    if 'FFN_1' in config.mpo_layers:
        if 'FFN_1' not in config.load_layer:
            for i in range(12):
                model.bert.encoder.layer[i].intermediate.from_pretrained_mpo()
        for i in range(12):
            del model.bert.encoder.layer[i].intermediate.dense
        
    if 'FFN_2' in config.mpo_layers:
        if 'FFN_2' not in config.load_layer:
            for i in range(12):
                model.bert.encoder.layer[i].output.from_pretrained_mpo()
        for i in range(12):
            del model.bert.encoder.layer[i].output.dense
    
    if 'word_embed' in config.mpo_layers:
        if 'word_embed' not in config.load_layer:
            model.bert.embeddings.from_pretrained_mpo()
        else:
            logger.info("Check load layer word_embed without from_pretrained...")
        del model.bert.embeddings.word_embeddings

    if 'attention' in config.mpo_layers:
        if 'attention' not in config.load_layer:
            for i in range(12):
                model.bert.encoder.layer[i].attention.self.from_pretrained_mpo()
                model.bert.encoder.layer[i].attention.output.from_pretrained_mpo()
        for i in range(12):   
            del model.bert.encoder.layer[i].attention.self.query
            del model.bert.encoder.layer[i].attention.self.key
            del model.bert.encoder.layer[i].attention.self.value
            del model.bert.encoder.layer[i].attention.output.dense

    if 'pooler' in config.mpo_layers:
        if 'pooler' not in config.load_layer:
            model.bert.pooler.from_pretrained_mpo()
        del model.bert.pooler.dense
    
    ########## moe setting
    if config.moe_type == 'moe' and training_args.do_train:
        logger.info("Check load from pretrained FFN weight ...")
        for i in range(config.n_layer):
            model.bert.encoder.layer[i].intermediate.moe_layer.from_pretrained_mpo(expert_new=model.bert.encoder.layer[i].intermediate.dense, use_mpo=False)
            del model.bert.encoder.layer[i].intermediate.dense
    elif config.moe_type == 'switch' and training_args.do_train:
        logger.info("Check load from pretrained FFN weight ...")
        for i in range(config.n_layer):
            model.bert.encoder.layer[i].intermediate.sffn.from_pretrained_mpo(expert_new=model.bert.encoder.layer[i].intermediate.dense, use_mpo=False)
            del model.bert.encoder.layer[i].intermediate.dense
    
    # tensor_set = model.encoder.block[0].layer[0].SelfAttention.q_mpo
    # mpo = MPO(model.encoder.block[0].layer[0].SelfAttention.q_mpo.mpo_input_shape, model.encoder.block[0].layer[0].SelfAttention.q_mpo.mpo_output_shape, 30000)
    # q_mpo_weight = mpo.mpo2matrix(tensor_set)

    # set num_beams for evaluation
    if data_args.eval_beams is None:
        data_args.eval_beams = model.config.num_beams

    # freezing the parameters.
    if training_args.do_train:
        freezing_params(model, training_args, model_args, adapter_args)

    if training_args.print_num_parameters:
        logger.info(model)
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info("Parameter name %s", name)
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info("Total trainable parameters %s", total_trainable_params)
        logger.info("Total parameters %s", total_params)
    # Gets the training/test/validation datasets.
    dataset_class = AutoTask
    if training_args.do_train:
        train_datasets = [dataset_class.get(task, seed=data_args.data_seed).get_dataset(
            split="train", n_obs=data_args.n_train, add_prefix=False if training_args.train_adapters else True)
            for task in data_args.tasks]
        dataset_sizes = [len(train_dataset) for train_dataset in train_datasets]
        train_dataset = datasets.concatenate_datasets(train_datasets)
    training_args.remove_unused_columns = False
    eval_datasets = ({task: dataset_class.get(task, seed=data_args.data_seed).get_dataset(
        split="validation", n_obs=data_args.n_val,
        add_prefix=False if training_args.train_adapters else True,
        split_validation_test=training_args.split_validation_test)
                         for task in data_args.eval_tasks}
                     if training_args.do_eval or training_args.evaluation_strategy != EvaluationStrategy.NO
                     else None)
    test_dataset = (
        {task: dataset_class.get(task, seed=data_args.data_seed).get_dataset(
            split="test", n_obs=data_args.n_test,
            add_prefix=False if training_args.train_adapters else True,
            split_validation_test=training_args.split_validation_test)
            for task in data_args.eval_tasks} if training_args.do_test else None
    )
    # Defines the metrics for evaluation.
    compute_metrics_fn = (
        build_compute_metrics_fn(data_args.eval_tasks, tokenizer) if training_args.predict_with_generate else None
    )
    # Defines the trainer.
    trainer = BERTTrainer(
        model=model,
        config=config,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=test_dataset,
        data_collator=TaskCollator(tokenizer, data_args, tpu_num_cores=training_args.tpu_num_cores),
        compute_metrics=None,
        multi_task_compute_metrics=compute_metrics_fn,
        data_args=data_args,
        dataset_sizes=dataset_sizes if training_args.do_train else None,
        adapter_config=adapter_config
    )
    if trainer.is_world_process_zero():
        arguments = get_training_args([model_args, data_args, training_args, adapter_args])
        handle_metrics("arguments", arguments, training_args.output_dir)

    # Trains the model.
    if training_args.do_train:
        if trainer.is_world_process_zero():
           last_checkpoint_path = training_args.output_dir
           model_path = model_args.model_name_or_path if (training_args.optimize_from_scratch or not os.path.exists(os.path.join(last_checkpoint_path, 'pytorch_model.bin')))\
             else last_checkpoint_path
        if training_args.compute_time:
           torch.cuda.synchronize()  # wait for move to complete
           start = torch.cuda.Event(enable_timing=True)
           end = torch.cuda.Event(enable_timing=True)
           start.record()
        trainer.train(
            #get_last_checkpoint_path(training_args.output_dir) \
            model_path=model_path \
                if (os.path.exists(training_args.output_dir) and not training_args.optimize_from_scratch) else None,
        )
        if training_args.compute_time: 
           torch.cuda.synchronize()  # wait for all_reduce to complete
           end.record()
           total_time = {"total_time": start.elapsed_time(end)}
           print("###### total_time ", total_time)
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_process_zero():
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))
            tokenizer.save_pretrained(training_args.output_dir)
     
    # Evaluation
    all_metrics = {}
    if training_args.do_eval or training_args.do_test:
        if trainer.is_world_process_zero():
            # By default we load  the model from last checkpoint path,
            # in case of saving the model with the best metrics, make sure to
            # set save_total = 1 so the best model is loaded here.
            # if not exists returns the path to the output_dir.
            if config.load_experiment:
                logger.info("Check using load_experiment: {}".format(config.load_experiment))
                last_checkpoint_path = get_last_checkpoint_path(config.load_experiment)
            else:
                logger.info("Check using output_dir: {}".format(training_args.output_dir))   
                last_checkpoint_path = get_last_checkpoint_path(training_args.output_dir)            
            config = AutoConfig.from_pretrained(
                last_checkpoint_path,
                cache_dir=model_args.cache_dir)
            # use_checkpoint_weight = True
            # if hasattr(config, 'mpo_layers'):
            #     # if config loaded by "from_pretrained" has mpo_layers, good weights are mpo weight. Or, good weights are original weights 
            #     use_checkpoint_weight = False
            for k, v in custom_config.items():
                setattr(config, k, v)
            config.batch_size = training_args.per_device_train_batch_size
            config.max_seq_length = data_args.max_source_length
            
            model = BertForSequenceClassification.from_pretrained(
                last_checkpoint_path,
                from_tf=".ckpt" in training_args.output_dir,
                config=config,
                cache_dir=model_args.cache_dir,
                adapter_config=adapter_config
            )
            if not training_args.do_train:
                if 'word_embed' in config.mpo_layers:
                    if 'word_embed' not in config.load_layer:
                        model.from_pretrained_mpo()
                    del model.shared
                    del model.lm_head
                if 'mlp' in config.mpo_layers or 'attention' in config.mpo_layers:
                    for i in range(config.num_layers):
                        model.encoder.block[i].from_pretrained_mpo()
                        model.decoder.block[i].from_pretrained_mpo()
                        if config.moe_type == 'moe':
                            logger.info("Check using moe mlp mpo")
                            model.encoder.block[i].layer[1].moe_layer.from_pretrained_mpo(expert_new=model.encoder.block[i].layer[1].DenseReluDense, use_mpo=True)
                            model.decoder.block[i].layer[2].moe_layer.from_pretrained_mpo(expert_new=model.decoder.block[i].layer[2].DenseReluDense, use_mpo=True)
                            
                            del model.encoder.block[i].layer[1].DenseReluDense
                            del model.encoder.block[i].layer[1].moe_layer.experts.w1
                            del model.encoder.block[i].layer[1].moe_layer.experts.w2
                            del model.decoder.block[i].layer[2].DenseReluDense
                            del model.decoder.block[i].layer[2].moe_layer.experts.w1
                            del model.decoder.block[i].layer[2].moe_layer.experts.w2
                        elif config.moe_type == 'switch':
                            logger.info("Check using switch mlp mpo")
                            model.encoder.block[i].layer[1].sffn.from_pretrained_mpo(expert_new=model.encoder.block[i].layer[1].DenseReluDense, use_mpo=True)
                            model.decoder.block[i].layer[2].sffn.from_pretrained_mpo(expert_new=model.decoder.block[i].layer[2].DenseReluDense, use_mpo=True)
                            
                            del model.encoder.block[i].layer[1].DenseReluDense
                            del model.decoder.block[i].layer[2].DenseReluDense
                        model.encoder.block[i].clear_ori_weight()   
                        model.decoder.block[i].clear_ori_weight()
                elif config.moe_type == 'moe':
                    logger.info("Check using moe mlp mpo")
                    for i in range(config.num_layers):
                        model.encoder.block[i].layer[1].moe_layer.from_pretrained_mpo(expert_new=model.encoder.block[i].layer[1].DenseReluDense, use_mpo=False)
                        model.decoder.block[i].layer[2].moe_layer.from_pretrained_mpo(expert_new=model.decoder.block[i].layer[2].DenseReluDense, use_mpo=False)

                        del model.encoder.block[i].layer[1].DenseReluDense
                        del model.decoder.block[i].layer[2].DenseReluDense
                elif config.moe_type == 'switch':
                    logger.info("Check using switch mlp mpo")
                    for i in range(config.num_layers):
                        model.encoder.block[i].layer[1].sffn.from_pretrained_mpo(expert_new=model.encoder.block[i].layer[1].DenseReluDense, use_mpo=False)
                        model.decoder.block[i].layer[2].sffn.from_pretrained_mpo(expert_new=model.decoder.block[i].layer[2].DenseReluDense, use_mpo=False)

                        del model.encoder.block[i].layer[1].DenseReluDense
                        del model.decoder.block[i].layer[2].DenseReluDense

            # NOTE: if trainer is not re-defined, there is a bug in the codes, that making
            # huggingface codes does not using the best checkpoint.
            trainer = BERTTrainer(
                model=model,
                config=config,
                args=training_args,
                train_dataset=train_dataset if training_args.do_train else None,
                eval_dataset=eval_datasets,
                data_collator=TaskCollator(tokenizer, data_args, tpu_num_cores=training_args.tpu_num_cores),
                compute_metrics=None,
                multi_task_compute_metrics=compute_metrics_fn,
                data_args=data_args,
                dataset_sizes=dataset_sizes if training_args.do_train else None,
                adapter_config=adapter_config
            )

        if training_args.train_adapters:
            if adapter_args.adapter_config_name == "adapter" and data_args.adapters is not None:
                for name, sub_module in model.named_modules():
                    task_to_adapter = {eval_task: adapter for eval_task, adapter in
                                       zip(data_args.eval_tasks, data_args.adapters)}
                    if isinstance(sub_module, AdapterController):
                        sub_module.set_task_to_adapter_map(task_to_adapter)

    if training_args.do_eval:
        metrics = trainer.evaluate()
        if trainer.is_world_process_zero():
            handle_metrics("val", metrics, training_args.output_dir)
            all_metrics.update(metrics)

    if training_args.do_test:
        metrics = trainer.evaluate(test_dataset)
        if trainer.is_world_process_zero():
            handle_metrics("test", metrics, training_args.output_dir)
            all_metrics.update(metrics)

    if torch.cuda.is_available() and training_args.compute_memory:
        peak_memory = torch.cuda.max_memory_allocated()/1024**2
        print(
            "Memory utilization",
            peak_memory,
            "MB"
        )
        memory_usage = {"peak_memory": peak_memory}
    return all_metrics


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
