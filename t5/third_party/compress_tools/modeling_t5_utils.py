import copy
from attr import has
import torch
import warnings
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers import logging
from transformers.models.t5.configuration_t5 import T5Config
from transformers.modeling_utils import find_pruneable_heads_and_indices, prune_linear_layer
from transformers.models.t5.modeling_t5 import T5LayerNorm
from transformers.activations import ACT2FN
from MPOE.adapters import (AutoAdapterController, MetaAdapterConfig,
                              TaskEmbeddingController, LayerNormHyperNet,
                              AdapterLayersHyperNetController,
                              MetaLayersAdapterController,
                              AdapterLayersOneHyperNetController)
import math
from functools import reduce
from MPOE.third_party.compress_tools.mixture_of_experts import MoE
from labml_helpers.module import M, TypedModuleList


##### mpo_lab
# from mpo_lab.MPOtorch import LinearDecomMPO, MPOattention
from .MPOtorch import LinearDecomMPO
from mpo_lab.Matrix2MPO import MPO
logger = logging.get_logger(__name__)

def check_shape(input_shape, output_shape):
    '''
    input_shape: (3072,768) or ([4,4,8,6,4], [4,4,4,4,3])
    output_shape: (3072,768) or ([4,4,8,6,4], [4,4,4,4,3])
    '''
    def calc_product(li):
        return reduce(lambda x,y: x*y,li)
    if isinstance(input_shape[0], list):
        input_shape[0] = calc_product(input_shape[0])
    if isinstance(input_shape[1], list):
        input_shape[1] = calc_product(input_shape[1])
    if isinstance(output_shape[0], list):
        output_shape[0] = calc_product(output_shape[0])
    if isinstance(output_shape[1], list):
        output_shape[1] = calc_product(output_shape[1])
    
    assert input_shape==output_shape, "Check mpo input shape != weight input shape"

def clone_module_list(module: M, n: int) -> TypedModuleList[M]:
    """
    ## Make a `nn.ModuleList` with clones of a given layer
    """
    return TypedModuleList([copy.deepcopy(module) for _ in range(n)])
class SwitchFFN(nn.Module):
    def __init__(self, *, config, expert, n_experts, d_model, capacity_factor=1.0, drop_tokens=False, tensor_learn=False, train_AT_index=2, is_scale_prob=False):
        super().__init__()
        self.n_experts = n_experts
        if config.switch_dropout > 0.0:
            logger.info("Check using switch dropout: {}".format(config.switch_dropout))
            expert.dropout = nn.Dropout(config.switch_dropout)
        self.experts = clone_module_list(expert, n_experts)

        self.switch = nn.Linear(d_model, n_experts)
        self.softmax = nn.Softmax(dim=-1)
        self.capacity_factor = capacity_factor
        self.drop_tokens = drop_tokens
        self.is_scale_prob = is_scale_prob
        self.config = config
    def forward(self, x):
        # x,(batch_size, length, dim)
        batch_size, seq_len, d_model = x.shape
        x = x.view(-1,d_model) # 直接去掉batch的概念，看成一个超长的token串

        route_prob = self.softmax(self.switch(x))
        route_prob_max, routes = torch.max(route_prob, dim=-1)
        if self.is_scale_prob:
            logger.info("Check is_scale_prob=True")
            factor = route_prob_max
        # Don't scale the values but multiply by $\frac{p}{\hat{p}} = 1$ so that the gradients flow
        else:
            factor = route_prob_max / route_prob_max.detach()
        x = x * factor.view(-1, 1)
        # Get indexes of tokens going to each expert
        indexes_list = [torch.eq(routes, i).nonzero(as_tuple=True)[0] for i in range(self.n_experts)] # 表示0号expert拿的是4号token，1号expert拿的是[0,1]号token ，...，4号expert拿的是1号token，共有n_expert个元素

        # Initialize an empty tensor to store outputs
        final_output = x.new_zeros(x.shape)

        capacity = int(self.capacity_factor * len(x) / self.n_experts)
        # Number of tokens routed to each expert.
        counts = x.new_tensor([len(indexes_list[i]) for i in range(self.n_experts)])

        # Initialize an empty list of dropped tokens
        dropped = []
        # Only drop tokens if `drop_tokens` is `True`.
        if self.drop_tokens:
            # Drop tokens in each of the experts
            for i in range(self.n_experts):
                # Ignore if the expert is not over capacity
                if len(indexes_list[i]) <= capacity:
                    continue
                # Shuffle indexes before dropping
                indexes_list[i] = indexes_list[i][torch.randperm(len(indexes_list[i]))]
                # Collect the tokens over capacity as dropped tokens
                dropped.append(indexes_list[i][capacity:])
                # Keep only the tokens upto the capacity of the expert
                indexes_list[i] = indexes_list[i][:capacity]
        # Get outputs of the expert FFNs
        # -------先重建再前向
        # route_outputs = [self.experts[i](x[indexes_list[i], :]) for i in range(self.n_experts)]
        # route_outputs = [self.experts[i](x[indexes_list[i], :].reshape((batch_size, seq_len, d_model))) for i in range(self.n_experts)]
        route_outputs = []
        for i in range(self.n_experts):
            if len(indexes_list[i]) > 0:
                route_outputs.append(self.experts[i](x[indexes_list[i], :]))
            else:
                route_outputs.append([])
        # -------每一个expert对所有的token都做同样的处理
        

        # Assign to final output 排序到原始的位置
        for i in range(self.n_experts):
            if len(route_outputs[i])>0:
                final_output[indexes_list[i], :] = route_outputs[i].view(-1,d_model)

        # Pass through the dropped tokens
        if dropped:
            dropped = torch.cat(dropped)
            final_output[dropped, :] = x[dropped, :]

        # Change the shape of the final output back to `[seq_len, batch_size, d_model]`
        final_output = final_output.view(batch_size, seq_len, d_model)

        return final_output, counts, route_prob.sum(0), len(dropped)
    def from_pretrained_mpo(self, expert_new=None, use_mpo=False):
        # --- 直接用权重，但是发现这样不直接
        # self.experts1 = clone_module_list(expert1, self.n_experts)
        # self.experts2 = clone_module_list(expert2, self.n_experts)
        # self.experts3 = clone_module_list(expert3, self.n_experts)
        # self.experts4 = clone_module_list(expert4, self.n_experts)
        # self.expert_c = expert_c
        # --- 直接把MLP类作为一个独立的个体
        if use_mpo:
            if expert_new:
                self.experts = clone_module_list(expert_new, self.n_experts)
            shared_wi_ct = self.experts[0].wi_mpo.tensor_set[2]
            shared_wo_ct = self.experts[0].wo_mpo.tensor_set[2]

            for i in range(self.n_experts):
                self.experts[i].wi_mpo.tensor_set[2] = shared_wi_ct
                self.experts[i].wo_mpo.tensor_set[2] = shared_wo_ct
                if self.config.tensor_learn:
                    self.experts[i].wi_mpo.tensor_set[2].requires_grad = False
                    self.experts[i].wo_mpo.tensor_set[2].requires_grad = False
            logger.info("Check init switch FFN finished ...")
        else:
            if expert_new:
                self.experts = clone_module_list(expert_new, self.n_experts)

class EmbeddingMPO(nn.Module):
    '''
    use MPO decompose word embedding
    '''
    def __init__(self, num_embeddings, embedding_dim, mpo_input_shape, mpo_output_shape, truncate_num, **kwargs):
        super(EmbeddingMPO, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.mpo = MPO(mpo_input_shape, mpo_output_shape, truncate_num)
        self.tensor_set = None
        # self.weight = None if self.tensor_set is None else self.tensor_set[0] # nn.Parameter 这个只是为了应付tie_weights

    def forward(self, input):
        weight_rebuild = self.mpo.mpo2matrix(self.tensor_set)[:32128]
        return F.embedding(input, weight_rebuild)

    def step_trunc(self, tensor_set):
        self.tensor_set = tensor_set

class LinearDecomHead(LinearDecomMPO):
    def forward(self, x):
        ori_shape=x.shape
        res = x.reshape(-1, x.shape[-1])
        weight_rebuild = self.mpo.mpo2matrix(self.tensor_set)[:32128]
        res = F.linear(res, weight_rebuild)        
        return res.view((tuple(ori_shape[:-1])+(-1,)))

class T5DenseReluDense(nn.Module):
    def __init__(self, config):
        super().__init__() # d_model=768, d_ff=3072
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.config = config
        self.input_shape = (config.batch_size * config.max_seq_length, config.hidden_size)
        if 'mlp' in config.mpo_layers:
            self.mpo_input_shape, self.mpo_output_shape = [4,4,8,6,4],[3,4,4,4,4] # t5-base
            # self.mpo_input_shape, self.mpo_output_shape = [4,4,8,8,4],[4,4,4,4,4] # t5-large
            self.wi_mpo = LinearDecomMPO(self.mpo_input_shape, self.mpo_output_shape, config.linear_trunc,tensor_learn=self.config.tensor_learn)
            self.wo_mpo = LinearDecomMPO(self.mpo_output_shape, self.mpo_input_shape, config.linear_trunc,tensor_learn=self.config.tensor_learn)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        if 'mlp' in self.config.mpo_layers:
            hidden_states = self.wi_mpo(hidden_states)
            hidden_states = F.relu(hidden_states)
            hidden_states = self.dropout(hidden_states)
            hidden_states = self.wo_mpo(hidden_states)
        else:
            hidden_states = self.wi(hidden_states)
            hidden_states = F.relu(hidden_states)
            hidden_states = self.dropout(hidden_states)
            hidden_states = self.wo(hidden_states)
        return hidden_states
    def from_pretrained_mpo(self):
        if 'mlp' in self.config.mpo_layers:
            # wi mpo decomposition
            mpo = MPO(self.mpo_input_shape, self.mpo_output_shape, self.config.linear_trunc)
            mpo_tensor_set,_,_ = mpo.matrix2mpo(self.wi.weight.data.cpu().numpy())
            mpo_pretrain_weight = [i.flatten() for i in mpo_tensor_set]
            self.wi_mpo.from_pretrained(self.input_shape, mpo_pretrain_weight, mpo_tensor_set, self.wi.bias)

            # wo mpo decomposition
            mpo = MPO(self.mpo_output_shape, self.mpo_input_shape, self.config.linear_trunc)
            mpo_tensor_set,_,_ = mpo.matrix2mpo(self.wo.weight.data.cpu().numpy())
            mpo_pretrain_weight = [i.flatten() for i in mpo_tensor_set]
            self.wo_mpo.from_pretrained(self.input_shape, mpo_pretrain_weight, mpo_tensor_set, self.wo.bias)
    def clear_ori_weight(self):
        if hasattr(self, "wi"):
            del self.wi
        if hasattr(self, "wo"):
            del self.wo

class T5Attention(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias

        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        self.config = config
        # MPO setting
        if 'attention' in config.mpo_layers:
            self.input_shape = (config.batch_size * config.max_seq_length, config.hidden_size)
            self.mpo_input_shape, self.mpo_output_shape, self.attention_trunc = [3,4,4,4,4],[4,4,4,4,3], config.attention_trunc # t5-base
            # self.mpo_input_shape, self.mpo_output_shape, self.attention_trunc = [4,4,4,4,4],[4,4,4,4,4], config.attention_trunc # t5-large
            self.q_mpo = LinearDecomMPO(self.mpo_input_shape, self.mpo_output_shape, self.attention_trunc, use_bias=False)
            self.k_mpo = LinearDecomMPO(self.mpo_input_shape, self.mpo_output_shape, self.attention_trunc, use_bias=False)
            self.v_mpo = LinearDecomMPO(self.mpo_input_shape, self.mpo_output_shape, self.attention_trunc, use_bias=False)
            self.o_mpo = LinearDecomMPO(self.mpo_output_shape, self.mpo_input_shape, self.attention_trunc, use_bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_heads, self.key_value_proj_dim, self.pruned_heads
        )
        # Prune linear layers
        self.q = prune_linear_layer(self.q, index)
        self.k = prune_linear_layer(self.k, index)
        self.v = prune_linear_layer(self.v, index)
        self.o = prune_linear_layer(self.o, index, dim=1)
        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.inner_dim = self.key_value_proj_dim * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_postion_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_postion_if_large = torch.min(
            relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_postion_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length):
        """ Compute binned relative position bias """
        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
        )
        relative_position_bucket = relative_position_bucket.to(self.relative_attention_bias.weight.device)
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            assert (
                len(past_key_value) == 2
            ), "past_key_value should have 2 past states: keys and values. Got {} past states".format(
                len(past_key_value)
            )
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        def shape(states):
            """  projection """
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        def unshape(states):
            """  reshape """
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """ projects hidden states correctly to key/query states """
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        if 'attention' in self.config.mpo_layers:
            query_states = shape(self.q_mpo(hidden_states))
        
            # get key/value states
            
            key_states = project(
                hidden_states, self.k_mpo, key_value_states, past_key_value[0] if past_key_value is not None else None
            )
            value_states = project(
                hidden_states, self.v_mpo, key_value_states, past_key_value[1] if past_key_value is not None else None
            )
        else:
            query_states = shape(self.q(hidden_states))
        
            # get key/value states
            
            key_states = project(
                hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
            )
            value_states = project(
                hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
            )

        # compute scores
        scores = torch.matmul(
            query_states, key_states.transpose(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, real_seq_length, key_length), device=scores.device, dtype=scores.dtype
                )
            else:
                position_bias = self.compute_bias(real_seq_length, key_length)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -seq_length:, :]

            if mask is not None:
                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

        scores += position_bias
        attn_weights = F.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = F.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        if 'attention' in self.config.mpo_layers:
            attn_output = self.o_mpo(attn_output)
        else:
            attn_output = self.o(attn_output)

        present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs
    def from_pretrained_mpo(self):
        if 'attention' in self.config.mpo_layers:
            mpo = MPO(self.mpo_input_shape, self.mpo_output_shape, self.config.attention_trunc)
            mpo_tensor_set, _, _ = mpo.matrix2mpo(self.q.weight.data.cpu().numpy())
            mpo_pretrain_weight = [i.flatten() for i in mpo_tensor_set]
            self.q_mpo.from_pretrained(self.input_shape, mpo_pretrain_weight, mpo_tensor_set, self.q.bias)

            mpo_tensor_set, _, _ = mpo.matrix2mpo(self.k.weight.data.cpu().numpy())
            mpo_pretrain_weight = [i.flatten() for i in mpo_tensor_set]
            self.k_mpo.from_pretrained(self.input_shape, mpo_pretrain_weight, mpo_tensor_set, self.k.bias)

            mpo_tensor_set, _, _ = mpo.matrix2mpo(self.v.weight.data.cpu().numpy())
            mpo_pretrain_weight = [i.flatten() for i in mpo_tensor_set]
            self.v_mpo.from_pretrained(self.input_shape, mpo_pretrain_weight, mpo_tensor_set, self.v.bias)

            mpo = MPO(self.mpo_output_shape, self.mpo_input_shape, self.config.attention_trunc)
            mpo_tensor_set, _, _ = mpo.matrix2mpo(self.o.weight.data.cpu().numpy())
            mpo_pretrain_weight = [i.flatten() for i in mpo_tensor_set]
            self.o_mpo.from_pretrained(self.input_shape, mpo_pretrain_weight, mpo_tensor_set, self.o.bias)
    def clear_ori_weight(self):
        del self.q
        del self.k
        del self.v
        del self.o

class T5LayerCrossAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.EncDecAttention = T5Attention(config, has_relative_attention_bias=False)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        head_mask=None,
        past_key_value=None,
        use_cache=False,
        query_length=None,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            head_mask=head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
        )
        layer_output = hidden_states + self.dropout(attention_output[0])
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs

class T5LayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False, adapter_config=None):
        super().__init__()
        self.SelfAttention = T5Attention(
            config, has_relative_attention_bias=has_relative_attention_bias,
            # is_bidirectional=not config.is_decoder
        )
        self.train_adapters = config.train_adapters
        if self.train_adapters:
            self.unique_hyper_net = True if isinstance(adapter_config, MetaAdapterConfig) and \
                                            (adapter_config.unique_hyper_net or
                                             adapter_config.efficient_unique_hyper_net) else False
            self.train_adapter_blocks = adapter_config.train_adapters_blocks and not self.unique_hyper_net
            if self.train_adapter_blocks:
                self.adapter_controller = AutoAdapterController.get(adapter_config)
                self.is_meta_adapter = True if isinstance(adapter_config, MetaAdapterConfig) else False
            elif self.unique_hyper_net:
                self.layer_hyper_net = MetaLayersAdapterController(adapter_config)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            position_bias=None,
            head_mask=None,
            past_key_value=None,
            use_cache=False,
            output_attentions=False,
            task=None,
            task_embedding=None,
            t5_block_adapters=None
    ):
        norm_x = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            norm_x,
            mask=attention_mask,
            position_bias=position_bias,
            head_mask=head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        y = attention_output[0]
        if self.train_adapters and self.train_adapter_blocks:
            y = self.adapter_controller(task if not self.is_meta_adapter else task_embedding, y)
        elif self.train_adapters and self.unique_hyper_net:
            y = self.layer_hyper_net(y, t5_block_adapters.self_attention)
        layer_output = hidden_states + self.dropout(y)
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs

# class T5DenseGatedGeluDense(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
#         self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
#         self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
#         self.config = config
#         if 'mlp' in config.mpo_layers:
#             self.input_shape = (config.batch_size * config.max_seq_length, config.hidden_size)
#             self.mpo_input_shape, self.mpo_output_shape = [3,4,4,4,4],[4,4,8,6,4]
#             check_shape((self.mpo_input_shape, self.mpo_output_shape), (config.d_model, config.d_ff))
#             self.wi_0_mpo = LinearDecomMPO(self.mpo_input_shape, self.mpo_output_shape, config.linear_trunc,tensor_learn=self.config.tensor_learn)
#             self.wi_1_mpo = LinearDecomMPO(self.mpo_input_shape, self.mpo_output_shape, config.linear_trunc,tensor_learn=self.config.tensor_learn)
#             self.wo_mpo = LinearDecomMPO(self.mpo_output_shape, self.mpo_input_shape, config.linear_trunc,tensor_learn=self.config.tensor_learn)
#         self.dropout = nn.Dropout(config.dropout_rate)
#         self.gelu_act = ACT2FN["gelu_new"]

#     def forward(self, hidden_states):
#         if 'mlp' in self.config.mpo_layers:
#             hidden_gelu = self.gelu_act(self.wi_0_mpo(hidden_states))
#             hidden_linear = self.wi_1_mpo(hidden_states)
#             hidden_states = hidden_gelu * hidden_linear
#             hidden_states = self.dropout(hidden_states)
#             hidden_states = self.wo_mpo(hidden_states)
#         else:
#             hidden_gelu = self.gelu_act(self.wi_0(hidden_states))
#             hidden_linear = self.wi_1(hidden_states)
#             hidden_states = hidden_gelu * hidden_linear
#             hidden_states = self.dropout(hidden_states)
#             hidden_states = self.wo(hidden_states)
#         return hidden_states
#     def from_pretrained_mpo(self):
#         if 'mlp' in self.config.mpo_layers:
#             # wi_0 mpo decomposition
#             mpo = MPO(self.mpo_input_shape, self.mpo_output_shape, self.config.linear_trunc)
#             mpo_tensor_set,_,_ = mpo.matrix2mpo(self.wi_0.weight.data.cpu().numpy())
#             mpo_pretrain_weight = [i.flatten() for i in mpo_tensor_set]
#             self.wi_0_mpo.from_pretrained(self.input_shape, mpo_pretrain_weight, mpo_tensor_set, self.wi_0.bias)

#             # wi_0 mpo decomposition
#             mpo = MPO(self.mpo_input_shape, self.mpo_output_shape, self.config.linear_trunc)
#             mpo_tensor_set,_,_ = mpo.matrix2mpo(self.wi_1.weight.data.cpu().numpy())
#             mpo_pretrain_weight = [i.flatten() for i in mpo_tensor_set]
#             self.wi_1_mpo.from_pretrained(self.input_shape, mpo_pretrain_weight, mpo_tensor_set, self.wi_1.bias)

#             # wi_0 mpo decomposition
#             mpo = MPO(self.mpo_output_shape, self.mpo_input_shape, self.config.linear_trunc)
#             mpo_tensor_set,_,_ = mpo.matrix2mpo(self.wo.weight.data.cpu().numpy())
#             mpo_pretrain_weight = [i.flatten() for i in mpo_tensor_set]
#             self.wo_mpo.from_pretrained(self.input_shape, mpo_pretrain_weight, mpo_tensor_set, self.wo.bias)

class T5LayerFF(nn.Module):
    def __init__(self, config, adapter_config=None):
        super().__init__()
        self.DenseReluDense = T5DenseReluDense(config)
        self.train_adapters = config.train_adapters
        if self.train_adapters:
            raise NotImplementedError
        self.config = config
        use_mpo = False
        if 'mlp' in config.mpo_layers:
            use_mpo = True
        if self.config.moe_type == 'switch':
            logger.info("Check use switch")
            self.sffn = SwitchFFN(
                expert=self.DenseReluDense,
                n_experts=config.num_experts, 
                d_model=config.d_model, 
                is_scale_prob=config.is_scale_prob,
                train_AT_index=config.train_AT_index,
                capacity_factor=config.capacity_factor,
                config = config)
        elif self.config.moe_type == 'moe':
            logger.info("Check use moe")
            if self.config.tensor_learn:
                logger.info("Check use tensor_learn moe")
            else:
                logger.info("Check without tensor_learn moe")
            self.moe_layer = MoE(
                config.d_model,
                config=self.config,
                num_experts=config.num_experts,
                hidden_dim=config.d_ff,
                activation=ACT2FN['relu'],
                pdropout=self.config.dropout_rate,
                use_mpo=use_mpo,
                tensor_learn=self.config.tensor_learn
            )

        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states, task=None, task_embedding=None, t5_block_adapters=None):
        norm_x = self.layer_norm(hidden_states)
        if self.config.moe_type == 'moe':
            y,_ = self.moe_layer(norm_x)
        elif self.config.moe_type == 'switch':
            y,_,_,_ = self.sffn(norm_x)
        else:
            y = self.DenseReluDense(norm_x)
        if self.train_adapters and self.train_adapters_blocks:
            y = self.adapter_controller(task if not self.is_meta_adapter else task_embedding, y)
        elif self.train_adapters and self.unique_hyper_net:
            y = self.layer_hyper_net(y, t5_block_adapters.feed_forward)
        layer_output = hidden_states + self.dropout(y)
        return layer_output

class T5Block(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False, adapter_config=None):
        super().__init__()
        self.adapter_config = adapter_config
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.config = config
        self.layer.append(T5LayerSelfAttention(config, \
                                               has_relative_attention_bias=has_relative_attention_bias,
                                               adapter_config=self.adapter_config))
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(config))
        self.layer.append(T5LayerFF(config, self.adapter_config))
        self.from_pretrained_mpo(force_init=True)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            position_bias=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            encoder_decoder_position_bias=None,
            head_mask=None,
            past_key_value=None,
            use_cache=False,
            output_attentions=False,
            return_dict=False,
            task=None,
            task_embedding=None,
            t5_block_adapters=None
    ):
        if past_key_value is not None:
            assert self.is_decoder, "Only decoder can use `past_key_values`"
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            error_message = "There should be {} past states. 2 (past / key)\
            for self attention.{} Got {} past key / value states".format(
                expected_num_past_key_values, "2 (past / key) for cross \
                attention" if expected_num_past_key_values == 4 else "", \
                len(past_key_value),
            )
            assert len(past_key_value) == expected_num_past_key_values, \
                error_message

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            head_mask=head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            task=task,
            task_embedding=task_embedding,
            t5_block_adapters=t5_block_adapters
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        # Keep self-attention outputs and relative position weights
        attention_outputs = self_attention_outputs[2:]

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                head_mask=head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions
            )
            hidden_states = cross_attention_outputs[0]
            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + \
                                          cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states, task=task,
                                       task_embedding=task_embedding,
                                       t5_block_adapters=t5_block_adapters)
        outputs = (hidden_states,)

        outputs = outputs + (present_key_value_state,) + attention_outputs
        return outputs  # hidden-states, present_key_value_states,
        # (self-attention weights), (self-attention position bias),
        # (cross-attention weights), (cross-attention position bias)
    def from_pretrained_mpo(self, force_init=False):
        self.force_init = force_init
        if 'attention' in self.config.mpo_layers and (self.force_init or 'attention' not in self.config.load_layer):
            logger.info("Check using attention mpo")
            assert isinstance(self.layer[0], T5LayerSelfAttention)
            self.layer[0].SelfAttention.from_pretrained_mpo()
            if self.is_decoder:
                assert isinstance(self.layer[1], T5LayerCrossAttention)
                self.layer[1].EncDecAttention.from_pretrained_mpo()
        if 'mlp' in self.config.mpo_layers and (force_init or 'mlp' not in self.config.load_layer):
            logger.info("Check using mlp mpo")
            if self.is_decoder:
                assert isinstance(self.layer[2], T5LayerFF)
                self.layer[2].DenseReluDense.from_pretrained_mpo()
            else:
                assert isinstance(self.layer[1], T5LayerFF)
                self.layer[1].DenseReluDense.from_pretrained_mpo()
    def clear_ori_weight(self):
        if 'attention' in self.config.mpo_layers and (self.force_init or 'attention' not in self.config.load_layer):
            self.layer[0].SelfAttention.clear_ori_weight()
            if self.is_decoder:
                self.layer[1].EncDecAttention.clear_ori_weight()
                if 'mlp' in self.config.mpo_layers and (self.force_init or 'mlp' not in self.config.load_layer) and not self.config.moe_type:
                    self.layer[2].DenseReluDense.clear_ori_weight()              
            elif not self.config.moe_type:
                if 'mlp' in self.config.mpo_layers and (self.force_init or 'mlp' not in self.config.load_layer):
                    self.layer[1].DenseReluDense.clear_ori_weight() 

