# -*- coding: utf-8 -*-
import numpy as np
from torch import nn as nn
import torch
import logging
import os
from compress_tools.Matrix2MPO import MPO
from torch.nn import functional as F
from torch import nn

os.environ['KMP_DUPLICATE_LIB_OK']='True'
logger = logging.getLogger(__name__)

def linear_act(x):
    return x
class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
    def forward(self, x):
        return x.contiguous().view((-1,)+self.shape[0])

class TransposeLayer(nn.Module):
    def __init__(self, *args):
        super(TransposeLayer, self).__init__()
    def forward(self, x):
        return torch.transpose(x,0,1)
class LinearDecomMPO(nn.Module):
    '''
    compress using MPO method
    ref: Compressing deep neural networks by matrix product operators
    '''
    def __init__(self, mpo_input_shape, mpo_output_shape, trunc_num,
        tensor_learn=False,
        CT_learn=False,
        use_bias = True,
        activation = None,
        bias_initializer = 'zeros',
        kernel_regularizer = None,
        bias_regularizer = None,
        activity_regularizer = None,
        kernel_constraint = None,
        bias_constraint = None,
        debug = False,
        init_seed = 11111986,
        use_dropout=False,
        use_layernorm=False,
        lora_config=None,
        *args,
        **kwargs
    ):
        super(LinearDecomMPO, self).__init__()
        self.trunc_num = trunc_num
        self.tensor_learn = tensor_learn
        self.CT_learn = CT_learn
        mpo_input_shape = np.array(mpo_input_shape)
        mpo_output_shape = np.array(mpo_output_shape)
        #mpo_ranks = np.array(mpo_ranks)
        self.mpo_input_shape = mpo_input_shape
        self.mpo_output_shape = mpo_output_shape
        ##self.mpo_ranks = mpo_ranks
        self.num_dim = mpo_input_shape.shape[0]  # length of the train
        self.use_bias = use_bias
        self.activation = activation
        self.kernel = None
        self.use_dropout = use_dropout
        self.use_layernorm = use_layernorm
        self.tensor_set = None

        self.debug = debug
        self.init_seed = init_seed
        self.lora_linear = False
        if lora_config:
            self.in_features, self.out_features = np.prod(mpo_input_shape), np.prod(mpo_output_shape)
            self.lora = LoRA((lora_config.r, self.in_features), (self.out_features,lora_config.r ), lora_config)
            self.lora_linear = True
    
    def _compute_adapted_weight(self, scaling=None):
        mpo = MPO(self.mpo_input_shape, self.mpo_output_shape, self.trunc_num)

        weight = mpo.mpo2matrix(self.tensor_set)
        # Merge the weights and mark it

        if self.lora.composition_mode == "scale":
            delta_w = self.lora.lora_B
        else:
            delta_w = self.lora.lora_B @ self.lora.lora_A
        weight = self.lora.com(weight, delta_w, scaling=scaling)

        return weight

    def build_model(self,input_shape, use_kernel=None, bias=None):
        num_inputs = int(np.prod(input_shape[1::]))

    
        total_length = torch.from_numpy(np.array(np.sum(self.mpo_input_shape * self.mpo_output_shape *
                              self.mpo_ranks[1:] * self.mpo_ranks[:-1])))
        if isinstance(use_kernel,torch.Tensor):
            self.kernel = torch.empty(size=(total_length,),requires_grad=True,device=torch.device('cuda'))
            self.kernel.data.copy_(use_kernel)
        else:
            self.kernel = torch.empty(size=(total_length,), requires_grad=True, device=torch.device('cuda'))
        self.kernel.contiguous()

        if self.use_bias:
            self.bias = torch.empty(torch.from_numpy(np.array(np.prod(self.mpo_output_shape))), requires_grad=True,
                                    device=torch.device('cuda'))
            if isinstance(bias, torch.Tensor):
                self.bias.data.copy_(bias)
            else:
                nn.init.constant_(self.bias, 0.)

        # Pre-calculate the indices, shapes and cores
        self.inds = np.zeros(self.num_dim).astype('int32')
        self.shapes = np.zeros((self.num_dim, 2)).astype('int32')
        self.cores = [None] * self.num_dim
        for k in range(self.num_dim - 1, -1, -1):
            # This is the shape of (m_k * r_{k+1}) * (r_k * n_k)
            self.shapes[k] = (self.mpo_input_shape[k] * self.mpo_ranks[k + 1],
                              self.mpo_ranks[k] * self.mpo_output_shape[k])
            # Note that self.cores store only the pointers to the parameter vector
            self.cores[k] = nn.Parameter(data=self.kernel[self.inds[k]:self.inds[k] + np.prod(self.shapes[k])])
            if 0 < k:  # < self.num_dim-1:
                self.inds[k - 1] = self.inds[k] + np.prod(self.shapes[k])
        if self.debug:
            print('self.shapes = ' + str(self.shapes))

        # Calculate and print the compression factor
        self.MPO_size = total_length
        self.full_size = (np.prod(self.mpo_input_shape) * np.prod(self.mpo_output_shape))
        self.compress_factor = 1. * self.MPO_size / self.full_size
        print('Compression factor = ' + str(self.MPO_size) + ' / ' \
              + str(self.full_size) + ' = ' + str(self.compress_factor))
    def get_weight(self):
        mpo = MPO(self.mpo_input_shape, self.mpo_output_shape, self.trunc_num)
        
        return mpo.mpo2matrix(self.tensor_set)
    def forward(self, x):
        ##################### use rebulid
        mpo = MPO(self.mpo_input_shape, self.mpo_output_shape, self.trunc_num)
        res = x.reshape(-1, x.shape[-1])
        if self.lora_linear:
            res = F.linear(res, self._compute_adapted_weight(),self.bias)
        else:
            res = F.linear(res, mpo.mpo2matrix(self.tensor_set),self.bias)
        ##################### use rebuild
        ori_shape=x.shape

        return res.view((tuple(ori_shape[:-1])+(-1,)))
        # return res
    def from_pretrained(self, input_shape,kernel_pretrain,tensor_set,bias=None,use_bias=True, device=None):
        if device:
            self.tensor_set = torch.nn.ParameterList([nn.Parameter(torch.from_numpy(i).to(device)) for i in tensor_set])
        else:
            self.tensor_set = torch.nn.ParameterList([nn.Parameter(torch.from_numpy(i)) for i in tensor_set])
        CT_index = int((len(self.tensor_set)-1)/2)
        if self.tensor_learn:
            # self.tensor_set[3].requires_grad = False
            self.tensor_set[CT_index].requires_grad = False
            # self.tensor_set[5].requires_grad = False
        elif self.CT_learn:
            for i in range(len(self.tensor_set)):
                self.tensor_set[i].requires_grad = False    
            self.tensor_set[CT_index].requires_grad = True    

        if use_bias:
            self.bias = bias
        else:
            logger.info("Check no bias")
            self.bias = None


    def step_trunc(self, tensor_set):
        self.tensor_set = tensor_set

class MPOattention(LinearDecomMPO):
    def __init__(self,*args, **kwargs):
        super(MPOattention, self).__init__(*args, **kwargs)
       
        self.num_attention_heads = 12
        self.hidden_size = 768
        self.use_bias = False
        logger.info("Check hard code with attention num 12 and hidden size 768")
    
    def forward(self, x):

        mpo = MPO(self.mpo_input_shape, self.mpo_output_shape, 192)
       
        if self.use_bias and not self.use_init_bias:
            ########### bias 
            input_ones = torch.cat((x, torch.ones_like(x)[:,:,0].unsqueeze(2)), dim=2) # 32,128,769
            
            part_1 = mpo.mpo2matrix(self.tensor_set).view(self.num_attention_heads, self.hidden_size, self.hidden_size)
            part_12 = torch.cat((part_1,self.part_2),dim=1)
            part_34 = torch.cat((self.part_3,self.part_4),dim=1)
            big_att = torch.cat((part_12,part_34),dim=2)
            res = torch.einsum('blh,ahj->balj',input_ones, big_att) # 32,12,128,768
            res = torch.einsum('balj,bkj->balk',res, input_ones)
            
        elif self.use_init_bias and self.use_bias:
            res = x
            res = torch.einsum('blh,ahj->balj',x, mpo.mpo2matrix(self.tensor_set).view(self.num_attention_heads, self.hidden_size, self.hidden_size)) # 32,12,128,768
            res = torch.einsum('balj,bkj->balk',res,x) + self.bias# 32,12,128,128
        else:
            res = x
            res = torch.einsum('blh,ahj->balj',x, mpo.mpo2matrix(self.tensor_set).view(self.num_attention_heads, self.hidden_size, self.hidden_size)) # 32,12,128,768
            res = torch.einsum('balj,bkj->balk',res,x)# 32,12,128,128
        

        return res

    def from_pretrained(self,input_shape,tensor_set,w_q=None, w_k_trans=None, bias_q=None, bias_k=None, use_bias=False):
        self.tensor_set = torch.nn.ParameterList([nn.Parameter(torch.from_numpy(i).cuda(), requires_grad=True) for i in tensor_set]) # 第一个维度是192
        if self.tensor_learn:
            self.tensor_set[2].requires_grad = False
        self.use_bias = use_bias
        self.use_init_bias = True
        ########### test bias
        if (not self.use_init_bias) and self.use_bias:
        
            self.part_2 = torch.einsum('bah,ahj->abj',bias_q.view(1,12,64),w_k_trans).cuda().detach() # 12,1,768
            self.part_3 = torch.einsum('jah,ahb->ajb',w_q, bias_k.view(12,64,1)).cuda().detach() # 12,768,1
            self.part_4 = torch.einsum('iah,ahj->aij',bias_q.view(1,12,64),bias_k.view(12,64,1)).cuda().detach()
        ## bias re-init
        elif self.use_bias and self.use_init_bias:
            logger.info("Check use init bias")
            self.bias = nn.Parameter(data=torch.zeros(12,128,128).cuda(), requires_grad=True)
        

class LinearDecomMPO_linear(LinearDecomMPO):
    def build_model(self,input_shape):
        num_inputs = int(np.prod(input_shape[1::]))

        # Check the dimensionality
        if np.prod(self.mpo_input_shape) != num_inputs:
            raise ValueError("The size of the input tensor (i.e. product "
                             "of the elements in mpo_input_shape) should "
                             "equal to the number of input neurons %d." % num_inputs)
        if self.mpo_input_shape.shape[0] != self.mpo_output_shape.shape[0]:  # 为什么这里只考虑第一个维度
            raise ValueError("The number of input and output dimensions "
                             "should be the same.")
        if self.mpo_ranks.shape[0] != self.mpo_output_shape.shape[0] + 1:  # 这里没看太明白
            raise ValueError("The number of the MPO-ranks should be "
                             "1 + the number of the dimensions.")
        if self.debug:
            print('mpo_input_shape = ' + str(self.mpo_input_shape))
            print('mpo_output_shape = ' + str(self.mpo_output_shape))
            print('mpo_ranks = ' + str(self.mpo_ranks))
        if self.use_bias:
            self.bias = torch.empty(torch.from_numpy(np.array(np.prod(self.mpo_output_shape))), requires_grad=True,device=torch.device('cuda'))
            nn.init.constant_(self.bias, 0.)

        self.model = nn.Sequential()
        for i,k in enumerate(range(self.num_dim - 1, -1, -1)):
            # This is the shape of (m_k * r_{k+1}) * (r_k * n_k)
            linear_shape = (self.mpo_input_shape[k] * self.mpo_ranks[k + 1],
                            self.mpo_ranks[k] * self.mpo_output_shape[k])
            self.model.add_module('reshape_to_{}_{}'.format(linear_shape[0],i), Reshape((linear_shape[0],)))
            self.model.add_module('linear_{}'.format(i), nn.Linear(linear_shape[0],linear_shape[1],bias=False))
            if self.use_dropout:
                print('use dropout')
                self.model.add_module('dropout_{}'.format(i), nn.Dropout(0.8))
            if self.use_layernorm:
                print('use layernorm')
                self.model.add_module('layernorm_{}'.format(i), nn.LayerNorm(linear_shape[1]))
            self.model.add_module('2reshape_to_{}_{}'.format(self.mpo_output_shape[k],i), Reshape((self.mpo_output_shape[k],)))
            self.model.add_module('transpose_{}'.format(i), TransposeLayer())

    def forward(self, x):
        ori_shape = x.shape
        res = x.reshape(-1, x.shape[-1])
        x_shape = res.shape[0]
        res =self.model(res)
        # res is of size o_1 x ... x o_d x batch_size # by Alexander
        res = torch.transpose(torch.reshape(res, (-1, x_shape)), 0, 1)

        if self.use_bias:
            res = torch.add(res, self.bias)
        if self.activation is not None:
            res = self.activation(res)

        return res.view((tuple(ori_shape[:-1]) + (-1,)))
    def init_param(self,func):
        for i in self.model:
            if isinstance(i, nn.Linear):
                func(i.weight)

    def from_pretrianed(self, input_shape,kernel_pretrain):
        for ind,i in enumerate(kernel_pretrain):
            linear_i = getattr(self.model,'linear_{}'.format(ind))
            getattr(self.model,'linear_{}'.format(ind)).weight.data.copy_(torch.from_numpy(kernel_pretrain[ind].reshape(
                linear_i.in_features,linear_i.out_features
            ).T).cuda())

class LinearDecomMPO_lora(LinearDecomMPO):
    
    def __init__(self, mpo_input_shape, mpo_output_shape, trunc_num,
        tensor_learn=False,
        lora_config=None,
        CT_learn=False,
        use_bias = True,
        activation = None,
        bias_initializer = 'zeros',
        kernel_regularizer = None,
        bias_regularizer = None,
        activity_regularizer = None,
        kernel_constraint = None,
        bias_constraint = None,
        debug = False,
        init_seed = 11111986,
        use_dropout=False,
        use_layernorm=False,
        *args,
        **kwargs
    ):
        super().__init__(mpo_input_shape, mpo_output_shape, trunc_num, tensor_learn, CT_learn=False, use_bias = True, activation = None,
                        bias_initializer = 'zeros', kernel_regularizer = None, bias_regularizer = None, activity_regularizer = None,
                        kernel_constraint = None, bias_constraint = None, debug = False, init_seed = 11111986, use_dropout=False,
                        use_layernorm=False, *args, **kwargs)
        if not lora_config:
            raise ValueError("lora_config must be set!")
        self.in_features, self.out_features = np.prod(mpo_input_shape), np.prod(mpo_output_shape)
        self.lora = LoRA((lora_config.r, self.in_features), (self.out_features, lora_config.r), lora_config)
    
    def _compute_adapted_weight(self, scaling=None):
        mpo = MPO(self.mpo_input_shape, self.mpo_output_shape, self.trunc_num)

        weight = mpo.mpo2matrix(self.tensor_set)
        # Merge the weights and mark it

        if self.lora.composition_mode == "scale":
            delta_w = self.lora.lora_B
        else:
            delta_w = self.lora.lora_B @ self.lora.lora_A
        weight = self.lora.com(weight, delta_w, scaling=scaling)

        return weight
    
    def forward(self, x):
        ##################### use rebulid
        
        res = x.reshape(-1, x.shape[-1])
        # res = F.linear(res, mpo.mpo2matrix(self.tensor_set),self.bias)
        res = F.linear(res, self._compute_adapted_weight() ,self.bias)
        ##################### use rebuild
        ori_shape=x.shape

        return res.view((tuple(ori_shape[:-1])+(-1,)))
