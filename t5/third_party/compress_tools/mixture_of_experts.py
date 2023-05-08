import torch
from torch import nn
import torch.nn.functional as F

import math
from inspect import isfunction
from transformers.activations import ACT2FN
from inspect import isfunction
import numpy as np
from mpo_lab.Matrix2MPO import MPO
from transformers.utils import logging

logger = logging.get_logger(__name__)
logger.setLevel(logging.INFO)

# constants

MIN_EXPERT_CAPACITY = 4

# helper functions

def default(val, default_val):
    default_val = default_val() if isfunction(default_val) else default_val
    return val if val is not None else default_val

def cast_tuple(el):
    return el if isinstance(el, tuple) else (el,)

# tensor related helper functions

def top1(t):
    values, index = t.topk(k=1, dim=-1)
    values, index = map(lambda x: x.squeeze(dim=-1), (values, index))
    return values, index

def cumsum_exclusive(t, dim=-1):
    num_dims = len(t.shape)
    num_pad_dims = - dim - 1
    pre_padding = (0, 0) * num_pad_dims
    pre_slice   = (slice(None),) * num_pad_dims
    padded_t = F.pad(t, (*pre_padding, 1, 0)).cumsum(dim=dim)
    return padded_t[(..., slice(None, -1), *pre_slice)]

# pytorch one hot throws an error if there are out of bound indices.
# tensorflow, in contrast, does not throw an error
def safe_one_hot(indexes, max_length):
    max_index = indexes.max() + 1
    return F.one_hot(indexes, max(max_index + 1, max_length))[..., :max_length]

def init_(t):
    dim = t.shape[-1]
    std = 1 / math.sqrt(dim)
    return t.uniform_(-std, std)

# activations

class GELU_(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

GELU = nn.GELU if hasattr(nn, 'GELU') else GELU_

# expert class

class Experts(nn.Module):
    def __init__(self,
        dim,
        config,
        num_experts = 16,
        hidden_dim = None,
        activation = GELU,
        pdropout = 0.0,
        use_mpo = False,
        tensor_learn = False):
        super().__init__()
        
        self.use_mpo = use_mpo
        self.tensor_learn = tensor_learn
        self.config = config
        self.num_experts = num_experts
        hidden_dim = default(hidden_dim, dim * 4)
        num_experts = cast_tuple(num_experts)

        w1 = torch.zeros(*num_experts, dim, hidden_dim)
        w2 = torch.zeros(*num_experts, hidden_dim, dim)
        b1 = torch.zeros(hidden_dim)
        b2 = torch.zeros(dim)

        w1 = init_(w1)
        w2 = init_(w2)

        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)
        
        if self.use_mpo:
            if dim == 768:
                mpo_input_shape_fc, mpo_output_shape_fc, truncate_num_fc = [4,4,8,6,4],[3,4,4,4,4], config.linear_trunc # 768*3072
                mpo_input_shape_proj, mpo_output_shape_proj, truncate_num_proj =  [3,4,4,4,4], [4,4,8,6,4], config.linear_trunc # 3072*768
            elif config.n_embd == 1024:
                mpo_input_shape_fc, mpo_output_shape_fc, truncate_num_fc = [4,4,8,8,4],  [4,4,4,4,4],config.linear_trunc # 1024*4096
                mpo_input_shape_proj, mpo_output_shape_proj, truncate_num_proj =  [4,4,4,4,4], [4,4,8,8,4],config.linear_trunc # 4096*1024
            else:
                raise NotImplementedError
            self.mpo_w1 = MPO(mpo_input_shape_fc, mpo_output_shape_fc, truncate_num_fc)
            w1_tensort_set = [self.mpo_w1.matrix2mpo(w1[i].squeeze(0).numpy())[0] for i in range(self.num_experts)]
            self.w1_mpo = [np.stack([w1_tensort_set[i][j] for i in range(self.num_experts)], axis=0) for j in range(5)]
            
            self.mpo_w2 = MPO(mpo_input_shape_proj, mpo_output_shape_proj, truncate_num_proj)
            w2_tensort_set = [self.mpo_w2.matrix2mpo(w2[i].squeeze(0).numpy())[0] for i in range(self.num_experts)]
            self.w2_mpo = [np.stack([w2_tensort_set[i][j] for i in range(self.num_experts)], axis=0) for j in range(5)]
            if 'mlp' in config.load_layer:
                self.convert_types()
        else:
            self.w2_mpo = None
            self.w1_mpo = None

        self.b1 = nn.Parameter(b1)
        self.b2 = nn.Parameter(b2)
        if not isfunction(activation):
            self.act = activation()
        else:
            self.act = activation
        
        self.dropout = nn.Dropout(pdropout)

    def forward(self, x):
        if self.use_mpo:
            ######## 下面这个需要调换
            # TODO: 和下面的调换
            # w1_rebuild = torch.stack([self.mpo_w1.mpo2matrix([self.w1_mpo[j][i] for j in range(5)]) for i in range(self.num_experts)],0)
            # w2_rebuild = torch.stack([self.mpo_w2.mpo2matrix([self.w2_mpo[j][i] for j in range(5)]) for i in range(self.num_experts)],0)
            # TODO：下面这个应该是正确的模型方法，moe情况下减少参数。上面方法有部分模型是基于它实现的
            w1_rebuild = torch.stack([self.mpo_w1.mpo2matrix([self.w1_mpo[0][i], self.w1_mpo[1][i], self.w1_mpo[2],
                                                              self.w1_mpo[3][i], self.w1_mpo[4][i]]) for i in range(self.num_experts)],0)
            w2_rebuild = torch.stack([self.mpo_w2.mpo2matrix([self.w2_mpo[0][i], self.w2_mpo[1][i], self.w2_mpo[2],
                                                              self.w2_mpo[3][i], self.w2_mpo[4][i]]) for i in range(self.num_experts)],0)
            ######## 上面这个需要调换
            hidden = torch.einsum('...nd,...hd->...nh', x, w1_rebuild) + self.b1
            hidden = self.act(hidden)
            out    = torch.einsum('...nh,...dh->...nd', hidden, w2_rebuild) + self.b2
        else:
            hidden = torch.einsum('...nd,...dh->...nh', x, self.w1) + self.b1
            hidden = self.act(hidden)
            out    = torch.einsum('...nh,...hd->...nd', hidden, self.w2) + self.b2
        return self.dropout(out)
    def convert_types(self):
        device = torch.cuda.current_device()
        for i in range(5):
            self.w1_mpo[i] = nn.Parameter(torch.from_numpy(self.w1_mpo[i]).to(device), requires_grad=True)
            self.w2_mpo[i] = nn.Parameter(torch.from_numpy(self.w2_mpo[i]).to(device), requires_grad=True)
        self.w1_mpo = nn.ParameterList(self.w1_mpo)
        self.w2_mpo = nn.ParameterList(self.w2_mpo)
        if self.tensor_learn:
            self.w1_mpo[2].requires_grad = False
            self.w2_mpo[2].requires_grad = False

    def from_pretrained_mpo(self, expert_new):
        logger.info("Check from_pretrained in Experts")
        device = torch.cuda.current_device()
        # 分配的数值对象会一直存在，但是变量名由于是在函数中则会消失
        ######## 下面这个需要调换
        # shared_c_fc_ct = expert_new.c_fc_mpo.tensor_set[2].clone().detach().cpu().numpy()
        # shared_c_proj_ct = expert_new.c_proj_mpo.tensor_set[2].clone().detach().cpu().numpy()
        
        # for i in range(self.num_experts):
        #     self.w1_mpo[2][i] = shared_c_fc_ct
        #     self.w2_mpo[2][i] = shared_c_proj_ct
        # TODO：下面的是正确的方法
        self.w1_mpo[2] = self.w1_mpo[2][0]
        self.w2_mpo[2] = self.w2_mpo[2][0]
        ######## 下面这个需要调换
        for j in [0,1,3,4]:
            for i in range(self.num_experts):
                self.w1_mpo[j][i] = expert_new.wi_mpo.tensor_set[j].clone().detach().cpu().numpy()
                self.w2_mpo[j][i] = expert_new.wo_mpo.tensor_set[j].clone().detach().cpu().numpy()

            self.w1_mpo[j] = nn.Parameter(torch.from_numpy(self.w1_mpo[j]).to(device), requires_grad=True)
            self.w2_mpo[j] = nn.Parameter(torch.from_numpy(self.w2_mpo[j]).to(device), requires_grad=True)
        
        self.w1_mpo[2] = nn.Parameter(torch.from_numpy(self.w1_mpo[2]).to(device), requires_grad=True)
        self.w2_mpo[2] = nn.Parameter(torch.from_numpy(self.w2_mpo[2]).to(device), requires_grad=True)
        self.w1_mpo = nn.ParameterList(self.w1_mpo)
        self.w2_mpo = nn.ParameterList(self.w2_mpo)

        if self.tensor_learn:
            for ind in self.config.train_AT_index:
                self.w1_mpo[ind].requires_grad = False
                self.w2_mpo[ind].requires_grad = False

# the below code is almost all transcribed from the official tensorflow version, from which the papers are written
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/research/moe.py

# gating network

class Top2Gating(nn.Module):
    def __init__(
        self,
        dim,
        num_gates,
        eps = 1e-9,
        outer_expert_dims = tuple(),
        second_policy_train = 'random',
        second_policy_eval = 'random',
        second_threshold_train = 0.2,
        second_threshold_eval = 0.2,
        capacity_factor_train = 1.25,
        capacity_factor_eval = 2.):
        super().__init__()

        self.eps = eps
        self.num_gates = num_gates
        self.w_gating = nn.Parameter(torch.randn(*outer_expert_dims, dim, num_gates))

        self.second_policy_train = second_policy_train
        self.second_policy_eval = second_policy_eval
        self.second_threshold_train = second_threshold_train
        self.second_threshold_eval = second_threshold_eval
        self.capacity_factor_train = capacity_factor_train
        self.capacity_factor_eval = capacity_factor_eval

    def forward(self, x, importance = None):
        *_, b, group_size, dim = x.shape
        num_gates = self.num_gates

        if self.training:
            policy = self.second_policy_train
            threshold = self.second_threshold_train
            capacity_factor = self.capacity_factor_train
        else:
            policy = self.second_policy_eval
            threshold = self.second_threshold_eval
            capacity_factor = self.capacity_factor_eval

        raw_gates = torch.einsum('...bnd,...de->...bne', x, self.w_gating)
        raw_gates = raw_gates.softmax(dim=-1)

        # FIND TOP 2 EXPERTS PER POSITON
        # Find the top expert for each position. shape=[batch, group]

        gate_1, index_1 = top1(raw_gates)
        mask_1 = F.one_hot(index_1, num_gates).float()
        density_1_proxy = raw_gates

        if importance is not None:
            equals_one_mask = (importance == 1.).float()
            mask_1 *= equals_one_mask[..., None]
            gate_1 *= equals_one_mask
            density_1_proxy *= equals_one_mask[..., None]
            del equals_one_mask

        gates_without_top_1 = raw_gates * (1. - mask_1)

        gate_2, index_2 = top1(gates_without_top_1)
        mask_2 = F.one_hot(index_2, num_gates).float()

        if importance is not None:
            greater_zero_mask = (importance > 0.).float()
            mask_2 *= greater_zero_mask[..., None]
            del greater_zero_mask

        # normalize top2 gate scores
        denom = gate_1 + gate_2 + self.eps
        gate_1 /= denom
        gate_2 /= denom

        # BALANCING LOSSES
        # shape = [batch, experts]
        # We want to equalize the fraction of the batch assigned to each expert
        density_1 = mask_1.mean(dim=-2)
        # Something continuous that is correlated with what we want to equalize.
        density_1_proxy = density_1_proxy.mean(dim=-2)
        loss = (density_1_proxy * density_1).mean() * float(num_gates ** 2)

        # Depending on the policy in the hparams, we may drop out some of the
        # second-place experts.
        if policy == "all":
            pass
        elif policy == "none":
            mask_2 = torch.zeros_like(mask_2)
        elif policy == "threshold":
            mask_2 *= (gate_2 > threshold).float()
        elif policy == "random":
            probs = torch.zeros_like(gate_2).uniform_(0., 1.)
            mask_2 *= (probs < (gate_2 / max(threshold, self.eps))).float().unsqueeze(-1)
        else:
            raise ValueError(f"Unknown policy {policy}")

        # Each sequence sends (at most?) expert_capacity positions to each expert.
        # Static expert_capacity dimension is needed for expert batch sizes
        expert_capacity = min(group_size, int((group_size * capacity_factor) / num_gates))
        expert_capacity = max(expert_capacity, MIN_EXPERT_CAPACITY)
        expert_capacity_f = float(expert_capacity)

        # COMPUTE ASSIGNMENT TO EXPERTS
        # [batch, group, experts]
        # This is the position within the expert's mini-batch for this sequence
        position_in_expert_1 = cumsum_exclusive(mask_1, dim=-2) * mask_1
        # Remove the elements that don't fit. [batch, group, experts]
        mask_1 *= (position_in_expert_1 < expert_capacity_f).float()
        # [batch, experts]
        # How many examples in this sequence go to this expert
        mask_1_count = mask_1.sum(dim=-2, keepdim=True)
        # [batch, group] - mostly ones, but zeros where something didn't fit
        mask_1_flat = mask_1.sum(dim=-1)
        # [batch, group]
        position_in_expert_1 = position_in_expert_1.sum(dim=-1)
        # Weight assigned to first expert.  [batch, group]
        gate_1 *= mask_1_flat

        position_in_expert_2 = cumsum_exclusive(mask_2, dim=-2) + mask_1_count
        position_in_expert_2 *= mask_2
        mask_2 *= (position_in_expert_2 < expert_capacity_f).float()
        mask_2_flat = mask_2.sum(dim=-1)

        position_in_expert_2 = position_in_expert_2.sum(dim=-1)
        gate_2 *= mask_2_flat
        
        # [batch, group, experts, expert_capacity]
        combine_tensor = (
            gate_1[..., None, None]
            * mask_1_flat[..., None, None]
            * F.one_hot(index_1, num_gates)[..., None]
            * safe_one_hot(position_in_expert_1.long(), expert_capacity)[..., None, :] +
            gate_2[..., None, None]
            * mask_2_flat[..., None, None]
            * F.one_hot(index_2, num_gates)[..., None]
            * safe_one_hot(position_in_expert_2.long(), expert_capacity)[..., None, :]
        )

        dispatch_tensor = combine_tensor.bool().to(combine_tensor)
        return dispatch_tensor, combine_tensor, loss

# plain mixture of experts

class MoE(nn.Module):
    def __init__(self,
        dim,
        config,
        num_experts = 16,
        hidden_dim = None,
        activation = nn.ReLU,
        second_policy_train = 'random',
        second_policy_eval = 'random',
        second_threshold_train = 0.2,
        second_threshold_eval = 0.2,
        capacity_factor_train = 1.25,
        capacity_factor_eval = 2.,
        loss_coef = 1e-2,
        experts = None,
        pdropout = 0.0,
        use_mpo = False,
        tensor_learn = False):
        super().__init__()

        self.num_experts = num_experts

        gating_kwargs = {'second_policy_train': second_policy_train, 'second_policy_eval': second_policy_eval, 'second_threshold_train': second_threshold_train, 'second_threshold_eval': second_threshold_eval, 'capacity_factor_train': capacity_factor_train, 'capacity_factor_eval': capacity_factor_eval}
        self.gate = Top2Gating(dim, num_gates = num_experts, **gating_kwargs)
        self.experts = default(experts, lambda: Experts(dim, config, num_experts = num_experts, hidden_dim = hidden_dim, activation = activation, pdropout = pdropout, use_mpo=use_mpo, tensor_learn=tensor_learn))
        self.loss_coef = loss_coef

    def forward(self, inputs, **kwargs):
        b, n, d, e = *inputs.shape, self.num_experts
        dispatch_tensor, combine_tensor, loss = self.gate(inputs)
        expert_inputs = torch.einsum('bnd,bnec->ebcd', inputs, dispatch_tensor)

        # Now feed the expert inputs through the experts.
        orig_shape = expert_inputs.shape
        expert_inputs = expert_inputs.reshape(e, -1, d)
        expert_outputs = self.experts(expert_inputs)
        expert_outputs = expert_outputs.reshape(*orig_shape)

        output = torch.einsum('ebcd,bnec->bnd', expert_outputs, combine_tensor)
        return output, loss * self.loss_coef
    def from_pretrained_mpo(self, expert_new=None, use_mpo=False):
        if expert_new and not use_mpo:
            w1_shape = torch.ones(len(self.experts.w1.shape),dtype=int)
            w1_shape[0] = self.num_experts
            w2_shape = torch.ones(len(self.experts.w2.shape),dtype=int)
            w2_shape[0] = self.num_experts
            b1_shape = torch.ones(len(self.experts.b1.shape),dtype=int)
            b1_shape[0] = self.num_experts
            b2_shape = torch.ones(len(self.experts.b2.shape),dtype=int)
            b2_shape[0] = self.num_experts

            self.experts.w1.data = expert_new.wi.weight.data.T.clone().repeat(*w1_shape)
            self.experts.w2.data = expert_new.wo.weight.data.T.clone().repeat(*w2_shape)
            # self.experts.b1.data = expert_new.wi.bias.data.clone()
            # self.experts.b2.data = expert_new.wo.bias.data.clone()
        elif expert_new and use_mpo:
            self.experts.from_pretrained_mpo(expert_new)

# 2-level heirarchical mixture of experts

class HeirarchicalMoE(nn.Module):
    def __init__(self,
        dim,
        num_experts = (4, 4),
        hidden_dim = None,
        activation = nn.ReLU,
        second_policy_train = 'random',
        second_policy_eval = 'random',
        second_threshold_train = 0.2,
        second_threshold_eval = 0.2,
        capacity_factor_train = 1.25,
        capacity_factor_eval = 2.,
        loss_coef = 1e-2,
        experts = None):
        super().__init__()

        assert len(num_experts) == 2, 'only 2 levels of heirarchy for experts allowed for now'
        num_experts_outer, num_experts_inner = num_experts
        self.num_experts_outer = num_experts_outer
        self.num_experts_inner = num_experts_inner

        gating_kwargs = {'second_policy_train': second_policy_train, 'second_policy_eval': second_policy_eval, 'second_threshold_train': second_threshold_train, 'second_threshold_eval': second_threshold_eval, 'capacity_factor_train': capacity_factor_train, 'capacity_factor_eval': capacity_factor_eval}

        self.gate_outer = Top2Gating(dim, num_gates = num_experts_outer, **gating_kwargs)
        self.gate_inner = Top2Gating(dim, num_gates = num_experts_inner, outer_expert_dims = (num_experts_outer,), **gating_kwargs)

        self.experts = default(experts, lambda: Experts(dim, num_experts = num_experts, hidden_dim = hidden_dim, activation = activation))
        self.loss_coef = loss_coef

    def forward(self, inputs, **kwargs):
        b, n, d, eo, ei = *inputs.shape, self.num_experts_outer, self.num_experts_inner
        dispatch_tensor_outer, combine_tensor_outer, loss_outer = self.gate_outer(inputs)
        expert_inputs_outer = torch.einsum('bnd,bnec->ebcd', inputs, dispatch_tensor_outer)

        # we construct an "importance" Tensor for the inputs to the second-level
        # gating.  The importance of an input is 1.0 if it represents the
        # first-choice expert-group and 0.5 if it represents the second-choice expert
        # group.  This is used by the second-level gating.
        importance = combine_tensor_outer.permute(2, 0, 3, 1).sum(dim=-1)
        importance = 0.5 * ((importance > 0.5).float() + (importance > 0.).float())

        dispatch_tensor_inner, combine_tensor_inner, loss_inner = self.gate_inner(expert_inputs_outer, importance = importance)
        expert_inputs = torch.einsum('ebnd,ebnfc->efbcd', expert_inputs_outer, dispatch_tensor_inner)

        # Now feed the expert inputs through the experts.
        orig_shape = expert_inputs.shape
        expert_inputs = expert_inputs.reshape(eo, ei, -1, d)
        expert_outputs = self.experts(expert_inputs)
        expert_outputs = expert_outputs.reshape(*orig_shape)

        # NOW COMBINE EXPERT OUTPUTS (reversing everything we have done)
        # expert_output has shape [y0, x1, h, d, n]

        expert_outputs_outer = torch.einsum('efbcd,ebnfc->ebnd', expert_outputs, combine_tensor_inner)
        output = torch.einsum('ebcd,bnec->bnd', expert_outputs_outer, combine_tensor_outer)
        return output, (loss_outer + loss_inner) * self.loss_coef
