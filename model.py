import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PatchFlatten(nn.Module) :
    def __init__(self, ch_size, img_size, p_size) :
        super(PatchFlatten , self).__init__()
        assert img_size % p_size == 0
        self.ch_size = ch_size
        self.img_size = img_size
        self.p_size = p_size

    def forward(self, img_tensor) :
        p_len = int(self.img_size / self.p_size)
        v_size = (self.p_size ** 2) * self.ch_size

        img_tensor = img_tensor.permute(0,2,3,1)
        img_patchs = torch.reshape(img_tensor , (-1, p_len, self.p_size, p_len, self.p_size, self.ch_size))
        patch_tensor = img_patchs.permute(0,1,3,2,4,5)

        v_tensor = torch.reshape(patch_tensor , (-1, p_len, p_len, v_size))
        v_tensor = torch.reshape(v_tensor , (-1, p_len**2, v_size))
        return v_tensor

class PositionEmbedding(nn.Module) :
    def __init__(self, p_len, v_size, em_dim, cuda_flag) :
        super(PositionEmbedding , self).__init__()
        self.p_len = p_len
        self.v_size = v_size
        self.em_dim = em_dim
        # start token 
        cls_tensor = torch.FloatTensor(np.random.randn(1,em_dim))
        if cuda_flag :
            cls_tensor = cls_tensor.cuda()
        self.cls_tensor = nn.Parameter(cls_tensor, requires_grad=True)
        # positional encoding tensor which is trainable
        pos_tensor = torch.FloatTensor(np.random.randn(1 , p_len+1 , em_dim))
        if cuda_flag :
            pos_tensor = pos_tensor.cuda()
        self.pos_tensor = nn.Parameter(pos_tensor , requires_grad=True) #(1 , patch_len+1 , em_dim)
        self.pos_linear = nn.Linear(v_size , em_dim)

    def forward(self, f_tensor) :
        batch_size = f_tensor.shape[0]
        # repeat cls tensor
        cls_tensor = self.cls_tensor.repeat(batch_size , 1) #(batch_size , em_dim)
        cls_tensor = cls_tensor.unsqueeze(1) #(batch_size , 1 , em_dim)
        # apply linear layer , convert vector shape patch vector size to embedding dimension
        x_tensor = self.pos_linear(f_tensor) #(batch_size , patch_len , em_dim)
        x_tensor = torch.cat([cls_tensor , x_tensor] , dim=1) #(batch_size , patch_len + 1 , em_dim)
        # add positional encoding to vector
        z_tensor = x_tensor + self.pos_tensor #(batch_size , patch_len+1 , em_dim)
        return z_tensor

class MultiHeadAttention(nn.Module) :
    def __init__(self, sen_size,  d_model, num_heads) :
        super(MultiHeadAttention , self).__init__()
        self.sen_size = sen_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = int(d_model / num_heads) # embedding_dim / num_heads

        self.q_layer = nn.Linear(d_model , d_model)
        self.k_layer = nn.Linear(d_model , d_model)
        self.v_layer = nn.Linear(d_model , d_model)
        self.o_layer = nn.Linear(d_model , d_model)

        self.scale = torch.sqrt(torch.tensor(self.depth , dtype=torch.float32 , requires_grad=False))

    def split(self , tensor) :
        tensor = torch.reshape(tensor , (-1 , self.sen_size , self.num_heads , self.depth)) # (batch_size , sen_size , num_heads , depth)
        tensor = torch.transpose(tensor , 1 , 2) # batch_size , num_heads , sen_size , depth)
        return tensor

    def merge(self , tensor) :
        tensor = torch.transpose(tensor , 1 , 2) # (batch_size , sen_size , num_heads , depth)
        tensor = torch.reshape(tensor , (-1 , self.sen_size , self.d_model)) # (batch_size , sen_size , embedding_dim)
        return tensor

    def scaled_dot_production(self, q_tensor, k_tensor, v_tensor, m_tensor) :
        q_tensor = self.split(q_tensor)
        k_tensor = self.split(k_tensor)
        v_tensor = self.split(v_tensor)
        k_tensor_T = torch.transpose(k_tensor , 2 , 3) # (batch_size , num_heads , depth , sen_size)

        qk_tensor = torch.matmul(q_tensor , k_tensor_T) # (batch_size , num_heads , sen_size , sen_size)
        qk_tensor /= self.scale
        if m_tensor != None :
            qk_tensor -= (m_tensor * 1e+6)

        qk_tensor = F.softmax(qk_tensor , dim = -1)
        att = torch.matmul(qk_tensor , v_tensor) # (batch_size , num_heads , sen_size , depth)
        return att

    def forward(self, q_in, k_in, v_in, m_in) :
        q_tensor = self.q_layer(q_in)
        k_tensor = self.k_layer(k_in)
        v_tensor = self.v_layer(v_in)

        att_tensor = self.scaled_dot_production(q_tensor, k_tensor, v_tensor, m_in)
        att_tensor = self.merge(att_tensor)

        o_tensor = self.o_layer(att_tensor)
        return o_tensor

class FeedForward(nn.Module) :
    def __init__(self, hidden_size, d_model) :
        super(FeedForward , self).__init__()
        self.hidden_size = hidden_size
        self.d_model = d_model
        self.ff = nn.Sequential(nn.Linear(d_model , hidden_size), 
                                nn.ReLU(),
                                nn.Linear(hidden_size , d_model))

    def forward(self, in_tensor) :
        o_tensor = self.ff(in_tensor)
        return o_tensor


class EncoderBlock(nn.Module) :
    def __init__(self, sen_size, d_model, num_heads, hidden_size, drop_rate, norm_rate) :
        super(EncoderBlock , self).__init__()
        self.sen_size = sen_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        self.mha_layer = MultiHeadAttention(sen_size , d_model , num_heads)
        self.ff_layer = FeedForward(hidden_size , d_model)

        self.drop1_layer = nn.Dropout(drop_rate)
        self.norm1_layer = nn.LayerNorm(d_model , eps=norm_rate)
        self.drop2_layer = nn.Dropout(drop_rate)
        self.norm2_layer = nn.LayerNorm(d_model , eps=norm_rate)

    def forward(self, in_tensor) :
        mha_tensor = self.mha_layer(in_tensor, in_tensor, in_tensor, None)
        mha_tensor = self.drop1_layer(mha_tensor)
        h_tensor = self.norm1_layer(in_tensor + mha_tensor) # residual connection

        ff_tensor = self.ff_layer(h_tensor)
        ff_tensor = self.drop2_layer(ff_tensor)
        o_tensor = self.norm2_layer(h_tensor + ff_tensor)
        return o_tensor

class Encoder(nn.Module) :
    def __init__(self, layer_size, sen_size, d_model, num_heads, hidden_size, drop_rate, norm_rate) :
        super(Encoder , self).__init__()
        self.layer_size = layer_size
    
        self.en_net = nn.Sequential()
        for i in range(layer_size) :
            en_block = EncoderBlock(sen_size, d_model, num_heads, hidden_size, drop_rate, norm_rate)
            self.en_net.add_module('Encoder_Block' + str(i) , en_block)

    def forward(self, in_tensor) :
        o_tensor = self.en_net(in_tensor)
        return o_tensor

class VIT(nn.Module) :
    def __init__(self , 
        layer_size , 
        class_size , 
        channel_size , 
        img_size , 
        patch_size , 
        em_dim , 
        num_heads , 
        hidden_size , 
        drop_rate , 
        norm_rate , 
        cuda_flag) :
        super(VIT , self).__init__()

        self.class_size = class_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.layer_size = layer_size
        self.em_dim = em_dim
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.drop_rate = drop_rate 
        self.norm_rate = norm_rate
        self.cuda_flag = cuda_flag
        

        p_len = int(img_size / patch_size) ** 2
        v_size = (patch_size ** 2) * channel_size
        self.p_flatten = PatchFlatten(channel_size, img_size, patch_size)
        self.p_embedding = PositionEmbedding(p_len, v_size, em_dim, cuda_flag)
        self.encoder = Encoder(layer_size, p_len+1, em_dim, num_heads, hidden_size, drop_rate, norm_rate)

        self.o_layer = nn.Linear(em_dim, class_size)

        self.init_param()

    # Xavier Initialization
    def init_param(self) :
        for p in self.parameters() :
            if p.dim() > 1 :
                nn.init.xavier_normal_(p)

    def forward(self, tensor) :
        f_tensor = self.p_flatten(tensor) # patch faltten
        em_tensor = self.p_embedding(f_tensor) # positional embedding
        o_tensor = self.encoder(em_tensor)

        index = torch.tensor([0])
        if self.cuda_flag == True :
            index = index.cuda()

        idx_tensor = torch.index_select(o_tensor, 1, index)
        idx_tensor = idx_tensor.squeeze(1)
        rep_tensor = self.o_layer(idx_tensor)
        return rep_tensor