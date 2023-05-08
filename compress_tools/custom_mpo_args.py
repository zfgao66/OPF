from dataclasses import dataclass, field
from typing import Callable, Dict, Optional


@dataclass
class CustomArguments:
    gpu_num : str = field(default='3',
                          metadata = {"help" : "which gpu to use"})
    mpo_lr: float = field(default=None,
                          metadata={"help": "mpo layer learning rate"})
    word_embed: bool = field(default=False,
                             metadata={"help": "whether use word_embed mpo"})
    mpo_layers : str = field(default='word_embed,attention',
                             metadata={"help" : "layers need to use mpo format"})
    emb_trunc : int = field(default=100000,
                            metadata={"help" : "truncate numbert of embedding"})
    linear_trunc: int = field(default=100000,
                              metadata={"help": "Truncate Rank of linear"})
    attention_trunc : int = field(default=100000,
                                  metadata={"help":"Truncate Rank of attention"})
    pooler_trunc : int = field(default=100000,
                               metadata={"help":"Truncate Rank of pooler"})
    load_layer : str = field(default='',
                             metadata={"help":"Layers which use to load"})
    update_mpo_layer : str = field(default='',
                               metadata={"help":"Layers which to update"})
    tensor_learn : bool = field(default=False,
                                metadata={"help":"Whether use tensor learn"})
    balance_attention : bool = field(default=False,
                            metadata={"help" : "Whether use [4,4,4,4,3]&[4,4,4,4,3] to replace [3,4,4,4,4]&[4,4,4,4,3]"})
    load_full : bool = field(default=False,
                            metadata={"help" : "Whether load full rank at first"})
    step_train : bool = field(default=False,
                            metadata={"help" : "Wheter use mask to clip parameters"})
    CT_learn : bool = field(default=False,
                            metadata={"help" : "True when only train CT"})
    runname : str = field(default="",
                            metadata={"help" : "Name of wandb"})
    share_former_layer_num : int = field(default=0) # delete after warmup
    share_layer : str = field(default="",
                            metadata={"help" : "Number of sharing layer index, e.g. 1,12 means range(1,12) have same CT with 0"})
    warmup : bool = field(default=False,
                            metadata={"help" : "**Only** load module of 'load_experiment' for initialize engine.module, without optimizer and lr_shceduler"})
    mpo_config : str = field(default="") # 配置mpo shape
    n_local : int = field(default=5) # mpo local tensors 长度
    stage2 : bool = field(default=False)
    load_from_albert : str = field(default="",
                                    metadata={"help" : "Albert path of loading"})
    classifier_dropout_prob : float = field(default=0.0,
                                            metadata={"help": "dropout probability of classifier"})
    num_hidden_groups : int = field(default=12)
    CT_lr : float = field(default=0.0)
    ### only for run_glue.py
    model_type : str = field(default="")
    lora_linear : bool = field(default=False)
    ### only for tune
    # do_hyper_tune : bool = field(default=False)
    ### for logging
    log_file_path : str = field(default="")