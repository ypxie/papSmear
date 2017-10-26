import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .torch_utils import *
from .local_utils import Indexflow, split_img, imshow
from collections import deque, OrderedDict
import functools

class passthrough(nn.Module):
    def __init__(self, **kwargs):
        super(passthrough, self).__init__()
    def forward(self, x, **kwargs):
        return x

class pretending_norm(nn.Module):
    def __init__(self, nchans, **kwargs):
        super(pretending_norm, self).__init__()
    def forward(self, x, **kwargs):
        return x

## Weights init function, DCGAN use 0.02 std
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1: 
        # Estimated variance, must be around 1
        m.weight.data.normal_(1.0, 0.02)
        # Estimated mean, must be around 0
        m.bias.data.fill_(0)

# def weights_init_selu(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1 or classname.find('Linear') != -1 :
#         if hasattr(m, 'weight'):
#             shape = list(m.weight.size()) # out, in, row, col
#             f_in = np.prod(shape[1::]) if len(shape) == 4 else shape[1] 
#             dev = np.sqrt(1.0 / f_in)
#             m.weight.data.normal_(0.0, dev) 

#     elif classname.find('BatchNorm') != -1:
#         # Estimated variance, must be around 1
#         m.weight.data.normal_(1.0, 0.02)
#         # Estimated mean, must be around 0
#         m.bias.data.fill_(0)

class sentConv(nn.Module):
    def __init__(self, in_dim, row, col, channel, norm,
                 activ = None, last_active = False):
        super(sentConv, self).__init__()
        self.__dict__.update(locals())
        out_dim = row*col*channel
        
        _layers = [nn.Linear(in_dim, out_dim)]
        
        _layers += [getNormLayer(norm, 1)(out_dim)]
        if last_active and  activ is not None:
            _layers += [activ] 
        
        self.out = nn.Sequential(*_layers)    
         
    def forward(self, inputs):
        linear_out = self.out(inputs)
        output = linear_out.view(-1, self.channel, self.row, self.col)
        #output = self.out(output)
        return output

def cat_vec_conv(text_enc, img_enc):
    # text_enc (B, dim)
    # img_enc  (B, chn, row, col)
    b, c = text_enc.size()
    row, col = img_enc.size()[2::]
    text_enc = text_enc.unsqueeze(-1).unsqueeze(-1)
    text_enc = text_enc.expand(b, c, row, col )
    com_inp = torch.cat([img_enc, text_enc], 1)
    return com_inp

class padConv2d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=1, stride=1, padding=None ,bias=False):
        super(padConv2d, self).__init__()

        if padding is None:
            left_row  = (kernel_size - 1) //2 
            right_row = (kernel_size - 1) - left_row
            left_col  = (kernel_size - 1) //2
            right_col = (kernel_size - 1) - left_col
            self.padding = (left_row, right_row, left_col, right_col)
        else:
            self.padding = (padding, padding, padding, padding)

        self.conv2d  = nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, 
                                 padding=0, bias=bias, stride=stride)

    def forward(self, inputs):
        padding = F.pad(inputs, self.padding, mode='reflect')
        output  = self.conv2d(padding)
        return output

class connectSide(nn.Module):
    def __init__(self, side_in, side_out, hid_in, sent_in, out_dim, 
                 norm, activ, down_rate, repeat= 0):
        # side_in is transformed to side_out, concate with hid_in, 
        # forward to down__rate smaller version concat with sent_in.
        # and use unet to preserve information.
        super(connectSide, self).__init__()
        self.__dict__.update(locals())

        _layers = []
        _layers += [nn.Conv2d(side_in, side_out, kernel_size = 1, padding=0, bias=True)]
        _layers += [getNormLayer(norm)(side_out )]
        _layers += [activ]
        self.side_trans = nn.Sequential(*_layers)
        
        _dict = OrderedDict()
        
        in_dim = side_out + hid_in

        in_list = [in_dim]
        for idx in range(down_rate):
            marker = 'down_{}'.format(idx)
            _dict[marker] = \
                conv_norm(in_dim,  out_dim, norm,  activ, 0, True,True,  3,None,2)
            in_dim = out_dim
            in_list.append(in_dim)

        for idx in range(down_rate):
            up_marker = 'up_{}'.format(idx)
            marker = 'conv_{}'.format(idx)
            _dict[up_marker] = nn.Upsample(scale_factor=2, mode='nearest')

            if idx != 0:
                in_dim = in_dim +  in_list[down_rate - idx]
            else:
                in_dim = in_dim + self.sent_in 

            _dict[marker] = \
                conv_norm(in_dim, out_dim, norm,  activ, 0, True,True,  3,None,1)
            in_dim = out_dim
        
        _dict['final_conv'] = \
                conv_norm(out_dim + in_list[0], out_dim, norm,  activ, 0, True,True,  1, 0,1)

        for k, v in _dict.items():
            setattr(self, k, v)   
            
    def forward(self, img_input, sent_input, hid_input):
        img_trans = self.side_trans(img_input)
        
        inputs = torch.cat( [img_trans, hid_input], 1)
        down_res_list = []
        for idx in range(self.down_rate):
            marker = 'down_{}'.format(idx)
            _layer = getattr(self, marker)
            down_res_list.append(inputs)
            _this_out = _layer(inputs)
            
            inputs = _this_out

        _conv_out = cat_vec_conv(sent_input, _this_out)
        
        for idx in range(self.down_rate):
            
            up_marker = 'up_{}'.format(idx)
            marker    = 'conv_{}'.format(idx)

            up_layer  = getattr(self, up_marker)
            conv_layer = getattr(self, marker)
            
            #print([a.size() for a in down_res_list])
            if idx != 0:
                _cap_inp = down_res_list[self.down_rate - idx]
                #print('cat size: ', _conv_out.size(), _cap_inp.size())
                _cat_out = torch.cat([_conv_out, _cap_inp], dim=1)
            else:
                _cat_out = _conv_out

            _up_out    = up_layer(_cat_out)
            _conv_out  = conv_layer(_up_out)
        
        _cat_out = torch.cat([_conv_out, down_res_list[0]], dim=1)
        final_out = getattr(self, 'final_conv')(_cat_out)

        return final_out

class connectSideBefore(nn.Module):
    def __init__(self, side_in, side_out, hid_in, sent_in, out_dim, 
                 norm, activ, up_rate, repeat= 0):
        # side_in is transformed to side_out, concate with upsampled sent_in, 

        super(connectSideBefore, self).__init__()
        self.__dict__.update(locals())

        _layers = []
        _layers += [nn.Conv2d(side_in, side_out, kernel_size = 1, padding=0, bias=True)]
        _layers += [getNormLayer(norm)(side_out )]
        _layers += [activ]
        self.side_trans = nn.Sequential(*_layers)
        
        _dict = OrderedDict()
        
        in_dim = sent_in
        _layers = []
        for idx in range(up_rate):
            sent_out = min(in_dim//2, 64)
            _layers += [nn.Upsample(scale_factor=2, mode='nearest')]
            _layers += [conv_norm(in_dim, sent_out, norm,  activ, 0, True,True, 3, None, 1)]
            in_dim = sent_out

        self.up_sent = nn.Sequential(*_layers)
        final_in_dim = sent_out + side_out  + hid_in

        self.final_conv = conv_norm(final_in_dim, out_dim, norm,  activ, 1, True,True,  3, 1, 1)
            
    def forward(self, img_input, sent_input, hid_input):
        img_trans = self.side_trans(img_input)
        up_sent = self.up_sent(sent_input)
        comp_input = torch.cat([img_trans, up_sent, hid_input], dim=1)
        final_out = self.final_conv(comp_input)

        return final_out


def up_conv(in_dim, out_dim, norm, activ, repeat=1, get_layer = False):
    _layers = [nn.Upsample(scale_factor=2,mode='nearest')]
    _layers += [padConv2d(in_dim,  in_dim, kernel_size = 3, stride=stride, bias=False)]
    _layers += [getNormLayer(norm)(out_dim )]
    _layers += [activ]

    for _ in range(repeat-1):
        _layers += [nn.Conv2d(out_dim,  out_dim,  kernel_size = 1, padding=0)]
        _layers += [getNormLayer(norm)(out_dim )]
        _layers += [activ]
    
    if get_layer:
        return nn.Sequential(*_layers)
    else:
        return _layers

def down_conv(in_dim, out_dim, norm, activ, repeat=1,
              kernel_size=3, get_layer = False):
    _layers = [padConv2d(in_dim,  out_dim, kernel_size = 3, stride=2, bias=False)]
    _layers += [getNormLayer(norm)(out_dim )]
    _layers += [activ]
    for _ in range(repeat):
        _layers += [nn.Conv2d(out_dim,  out_dim, kernel_size = 1, padding=0, bias=False)]
        _layers += [getNormLayer(norm)(out_dim )]
        _layers += [activ]
    if get_layer:
        return nn.Sequential(*_layers)
    else:
        return _layers

def conv_norm(in_dim, out_dim, norm, activ=None, repeat=1, get_layer = False,
              last_active=True, kernel_size=1, padding=None, stride=1, last_norm=True):
    _layers = []
    _layers += [padConv2d(in_dim,  out_dim, kernel_size = kernel_size,padding=padding, stride=stride, bias=False)]

    for _ in range(repeat):
        _layers += [getNormLayer(norm)(out_dim )]
        _layers += [activ] 
        _layers += [padConv2d(out_dim,  out_dim, kernel_size = kernel_size, padding=padding, bias=False)]
        
    if last_norm:
       _layers += [getNormLayer(norm)(out_dim )]

    if last_active and activ is not None:
       _layers += [activ] 
    if get_layer:
        return nn.Sequential(*_layers)
    else:
        return _layers

def brach_out(in_dim, out_dim, norm, activ, repeat= 1, get_layer = False):
    _layers = []
    for _ in range(repeat):
        _layers += [padConv2d(in_dim,  in_dim, kernel_size = 3, stride= 1, bias=False)]
        _layers += [getNormLayer(norm)(in_dim )]
        _layers += [activ]
    
    _layers += [nn.Conv2d(in_dim,  out_dim, 
                kernel_size = 1, padding=0, bias=False)]    
    _layers += [nn.Tanh()]

    if get_layer:
        return nn.Sequential(*_layers)
    else:
        return _layers


class catSentConv(nn.Module):
    def __init__(self, enc_dim, emb_dim, feat_size, 
                 norm, activ, down_rate):
        '''
          enc_dim: B*enc_dim*H*W
          emb_dim: the dimension of feeded embedding
          feat_size: the feature map size of the feature map. 
        '''
        super(catSentConv, self).__init__()
        self.__dict__.update(locals())

        inp_dim = enc_dim + emb_dim
        _layers = []
        #_layers =  conv_norm(inp_dim, enc_dim, norm, activ, 0, False, True, 1, 0, last_norm=False)
        for _ in range(down_rate):
            _layers +=  \
            conv_norm(inp_dim, enc_dim, norm, activ, 0, False, True, 3, 1, 2, last_norm=False)
            inp_dim = enc_dim

        new_feat_size = feat_size//int(2**down_rate)
        _layers += [nn.Conv2d(inp_dim, 1, kernel_size = new_feat_size, padding =0)]
        self.node = nn.Sequential(*_layers)

    def forward(self,sent_code,  img_code):
        sent_code =  sent_code.unsqueeze(-1).unsqueeze(-1)
        dst_shape = list(sent_code.size())
        #print(dst_shape, img_code.size())
        dst_shape[1] =  sent_code.size()[1]
        dst_shape[2] =  img_code.size()[2] 
        dst_shape[3] =  img_code.size()[3] 
        sent_code = sent_code.expand(dst_shape)
        #sent_code = sent_code.view(*dst_shape)
        #print(img_code.size(), sent_code.size())
        comp_inp = torch.cat([img_code, sent_code], dim=1)
        output = self.node(comp_inp)
        chn  = output.size()[1]
        output = output.view(-1, chn)

        return output

class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        input = to_contiguous(input).view(-1, input.size(2))
        target = to_contiguous(target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)
        output = - input.gather(1, target) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride = 1, kernel_size=3, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=1, # change
                               padding= (kernel_size-1)//2, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes , kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(planes )
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)

        return out

class LayerNorm1d(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm1d, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta  

class LayerNorm2d(nn.Module):
    ''' 2D Layer normalization module '''

    def __init__(self, features, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.gamma = nn.Parameter(torch.ones(1, features, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, features, 1, 1))
        self.eps = eps

    def forward(self, input):
        b,c,h,w = input.size()
        x = input.view(b, -1)
        mean = x.mean(-1, keepdim=True).view(b,1,1,1)
        std = x.std(-1, keepdim=True).view(b,1,1,1)

        out = self.gamma * (input -  mean)
        factor = (std + self.eps) + self.beta  
        out = out / factor
        return out

def getNormLayer(norm='bn', dim=2):

    norm_layer = None
    if dim == 2:
        if norm == 'bn':
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        elif norm == 'instance':
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
        elif norm == 'ln':
            norm_layer = functools.partial(LayerNorm2d)
    elif dim == 1:
        if norm == 'bn':
            norm_layer = functools.partial(nn.BatchNorm1d, affine=True)
        elif norm == 'instance':
            norm_layer = functools.partial(nn.InstanceNorm1d, affine=False)
        elif norm == 'ln':
            norm_layer = functools.partial(LayerNorm1d)
    assert(norm_layer != None)
    return norm_layer

def batch_forward(cls, BatchData, batch_size,**kwards):
    total_num = BatchData.shape[0]
    results = []
    for ind in Indexflow(total_num, batch_size, False):
        data = BatchData[ind]
        results.append(cls.forward(data, **kwards))
    return torch.cat(results, dim=0)

def split_testing(cls, inputs,  batch_size = 4, windowsize=None, testing= True,stateful=False):
    # since inputs is (B, T, C, Row, Col), we need to make (B*T*C, Row, Col)
    #windowsize = self.row_size
    board = 20
    adptive_batch_size=False # cause we dont need it for fixed windowsize.
    
    B, T, C, Row, Col = inputs.shape
    batched_imgs = inputs.reshape((B, T*C, Row, Col))
    outputs = np.zeros_like(inputs) # we don't consider multiple outputs

    for idx, img in enumerate(batched_imgs):  # this function only consider batch==1 case.
        PatchDict = split_img(img, windowsize = windowsize, board = board, fixed_window= True,step_size=None)
        output = None
        all_keys = PatchDict.keys()
        for this_size in all_keys:
            BatchData, org_slice_list, extract_slice_list = PatchDict[this_size]
            if adptive_batch_size == True:
                old_volume = batch_size * windowsize * windowsize
                new_bs = int(np.floor( 1.0*old_volume/np.prod(this_size)))
            else:
                new_bs = batch_size
            #print(new_bs, BatchData.shape[0])
            bat, time, rows, cols = BatchData.shape
            BatchData = BatchData.reshape((bat, time, C, rows, cols))
            thisprediction  =  batch_forward(cls, BatchData, batch_size, testing= testing,stateful=stateful)
            thisprediction  = thisprediction.cpu().numpy()
            #if type(thisprediction) != list:
            #    thisprediction = [thisprediction]
            #    thisprediction = [pred.cpu().numpy() for pred in thisprediction]
            if output is None:
                output = np.zeros((T, C, Row, Col))
            #[np.zeros( (B, T, C, Row, Col)) for _ in range(len(thisprediction))]
            #for odx, pred in enumerate(thisprediction) :
            for idx_, _ in enumerate(org_slice_list):
                org_slice = org_slice_list[idx_]
                extract_slice = extract_slice_list[idx_]
                output[: ,:, org_slice[0], org_slice[1]] = thisprediction[idx_][:,:,extract_slice[0], extract_slice[1]]

        outputs[idx] = output
    return outputs

def spatial_pool(x, keepdim=True):
    # input should be of N * channel * row * col
    x = torch.mean(x, -1, keepdim=False)
    x = torch.mean(x, -1, keepdim=False)
    if keepdim:
        return x.unsqueeze(-1).unsqueeze(-1)
    else:
        return x

class spatialAttention(nn.Module):
    # accept a query feat maps and memory feat maps
    # (b, len, c, row, col)
    # return memory_size output, gated by stride conv
    def __init__(self, query_chans, mem_chans, dilation_list=None, activ='selu', norm='bn'):
        super(spatialAttention, self).__init__()

        self.query_chans = query_chans
        self.mem_chans   = mem_chans
        self.conv_query  = ConvBN(query_chans, mem_chans, activ = activ, norm=norm)

        self.conv_ops = _make_nConv(query_chans + mem_chans, mem_chans, 
                                    depth=len(dilation_list[0:-1]), norm=norm, 
                                    activ = activ, dilation_list = dilation_list[0:-1] )
        self.final_conv = ConvBN(mem_chans, mem_chans, activ=activ, dilation=dilation_list[-1], 
                                 act= None, norm=norm)

    def forward(self, keys, mems):
        '''
        Parameters
        ---------
        keys: of size (b, c, row, col)
        mems: of size (b, c_mem, row, col)
        Returns:
        --------
        tensor of size (b, c_mem, row, col)
        ''' 
        inputs = torch.cat([keys, mems], dim=1)
        out_tensor = self.conv_ops(inputs)
        
        out_gate = F.sigmoid(self.final_conv(out_tensor))
        query_maps = self.conv_query(keys)
        
        output = out_gate * mems + (1-out_gate) * query_maps
        return output

def temperal_atten(querys, memory, strengths, atten_mask=None):
    # queries (batch_size, que_slot, query_size)
    # memory  (batch_size, mem_slot, query_size)
    # strength (bs, que_slot, 1)
    # return  (batch_size, query_size, mem_slot)
    distance = cosine_distance(memory, torch.transpose(querys,2,1).contiguous())
    #print('strengths size: ', strengths.size(), distance.size())
    strengths = torch.transpose(strengths, 1, 2).expand_as(distance)
    #distance (batch_size, mem_slot, que_slot)
    prob_dis = softmax(distance*strengths, 1)
    return torch.transpose(prob_dis, 2, 1).contiguous()

def position_encoding_init(n_position, d_pos_vec):
    ''' Init the sinusoid position encoding table '''
    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2*i/d_pos_vec) for i in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor)

class MultiHeadAttention(nn.Module):
    #''' Multi-Head Attention module '''
    def __init__(self, in_chans, output_chans, n_head, dilation_list, context_size = None, 
                 pos_emb_size = None, dropout=0.1, activ = 'selu', norm='bn'):
        super(MultiHeadAttention, self).__init__()
        self.max_pos_size = 100
        self.register_buffer('device_id', torch.zeros(1))
        self.n_head = n_head
        self.in_chans = in_chans
        self.pos_emb_size = output_chans if pos_emb_size is None else pos_emb_size

        self.output_chans = output_chans
        #self.compose_chans = output_chans + pos_emb_size

        #self.split_size = self.compose_chans//n_head # need to check whether mode==0
        self.conv_split_size = self.output_chans//n_head
        self.split_size = self.conv_split_size + pos_emb_size

        self.conv_mem   = nn.Conv2d(in_chans, output_chans, kernel_size= 3, padding=1, stride=1)
        self.conv_query = nn.Conv2d(in_chans, output_chans, kernel_size=3,  padding=1, stride=1)
        
        self.strength_transform = DenseStack(self.split_size, 64, 1, 1, activ=activ)
        self.dropout = nn.Dropout(dropout)
        #dilation_list  = [1, 2, 2, 4]
        
        self.spatial_atten = spatialAttention(in_chans, output_chans, dilation_list=dilation_list,
                                              activ=activ, norm=norm)
        self.context_size = context_size if context_size is not None else np.inf
        
        self.position_enc = nn.Embedding(self.max_pos_size, self.pos_emb_size, padding_idx = 0)
        self.position_enc.weight.data = position_encoding_init(self.max_pos_size, self.pos_emb_size)

        self.reset()
        
    def reset(self):
        self.mem_que = deque([], self.context_size)  # queue of tensor(bs, ch, row, col)
        self.pos_que = deque([], self.context_size)  # queue of tensor(1, self.split_size)
        
    def _update_mem_que(self, new_memory, org_mem_size, pos, testing=False):
        '''
        This function update self.mem_que using the new_memory
        -----------
        Parameters:
            new_memory (bs*len, outChan, row, col)
            org_mem_size is a tuple(bs, len, chn, row, col) the size of memory before conv
            pos: has to be torch int variable
        -----------
        Return:
            updated Mem_tensor (bs*len, chn, row, col)
        '''
        batch_size, len_mem, chn_mem, row, col = org_mem_size
        _, output_chans, _, _ = new_memory.size()
        
        memory = new_memory.view(batch_size, len_mem, output_chans, row, col)
        pos_emb = self.position_enc(pos)

        # we try to not include this memory.
        for idx in range(len_mem):
            self.mem_que.append(memory[:, idx])
            self.pos_que.append(pos_emb[idx])

        running_mem = torch.stack(self.mem_que, 1) # (bs, len, ch, row, col)
        running_pos = torch.stack(self.pos_que, 1) #(1, len, pos_emb_size)

        # we don't allow the memory to atten to current step except the first time step. Not a good idea?
        #if len(self.mem_que) > 1:
        #    self.updated_mem_len = len(self.mem_que) - 1
        #    running_mem = running_mem[:, 0:-1].contiguous()
        #    running_pos = running_pos[:,0:-1].contiguous()
        #else:

        self.updated_mem_len = len(self.mem_que)
        
        running_mem = running_mem.view(batch_size, self.updated_mem_len, output_chans, row, col)
        
        return running_mem, running_pos
    
    def get_pos_emb(self):
        pass

    def forward(self, query_maps, new_mem_maps, pos, eval=True,testing=False):
        '''
        we do attention for each time step seperately. so no need to mask anything.
        return (bs, nhead*split_size, row, col)
        ---------------------
        Parameters:
            query_maps (bs, len, chn, row, col) or (bs, chn, row, col) will add 1 dim automatically
            new_mem_maps (bs, len, chn, row, col) or (bs, chn, row, col) will add 1 dim automatically
            pos is a list of [1,2,3,4,5,6,7]
        Return:
            tensor of size (bd, output_chans, row, col)
        '''
        n_head, split_size = self.n_head, self.split_size

        if len(new_mem_maps.size()) == 4:
            new_mem_maps = new_mem_maps.unsqueeze(1)
        if len(query_maps.size()) == 4:
            query_maps = query_maps.unsqueeze(1)
        
        org_mem_size = new_mem_maps.size()

        batch_size, len_mem, chn_mem, row, col = org_mem_size
        batch_size, len_query, chn_query, row, col = query_maps.size()
        
        pos = [this_pos%self.context_size for this_pos in pos]
        len_pos = len(pos)
        pos = torch.from_numpy(np.asarray(pos).astype(np.int64)).unsqueeze(0)
        pos = to_device(pos, self.device_id, requires_grad=False)
        
        reshape_query  = query_maps.view(batch_size*len_query, chn_query, row, col)
        reshape_new_memory = new_mem_maps.view(batch_size*len_mem,  chn_mem, row, col)

        querys  = self.conv_query(reshape_query) # (b*len_que, outChans, row, col)
        this_memorys = self.conv_mem(reshape_new_memory)  # (b*len_mem, outChans, row, col)
        # update running memory queu
        # memory (bs, len, output_chans, row, col)
        # pos_emb (1, len, pos_emb_size)
        memory, mem_pos_emb = self._update_mem_que(this_memorys, org_mem_size, pos,testing)
        query_pos_emb = self.position_enc(pos) #(1,len, pos_emb_size)

        mem_pos_emb   = mem_pos_emb.expand(batch_size*n_head,   self.updated_mem_len, self.pos_emb_size)
        query_pos_emb = query_pos_emb.expand(batch_size*n_head, len_query, self.pos_emb_size)
        
        if eval:
            query_mean  = spatial_pool(querys).view(batch_size, len_query, self.output_chans)
            memory_mean = spatial_pool(memory).view(batch_size, self.updated_mem_len, self.output_chans)
            #compose pos_emb to mean
            #querys_mean_comp = torch.cat([query_mean,  query_pos_emb], -1) #(bs, len, compose_size)
            #mem_mean_comp    = torch.cat([memory_mean, mem_pos_emb], -1)     #(bs, len, compose_size)

            query_mean_split  = torch.split(query_mean,  split_size=self.conv_split_size, dim = -1)
            memory_mean_split = torch.split(memory_mean, split_size=self.conv_split_size, dim = -1)

            #this part need to be double checked n_head ahead or not?
            query_mean_split   = torch.stack(query_mean_split,  1).view(batch_size*n_head,  len_query,  self.conv_split_size) 
            memory_mean_split  = torch.stack(memory_mean_split, 1).view(batch_size*n_head,  self.updated_mem_len, self.conv_split_size) 
            
            query_mean_split_comp  = torch.cat([query_mean_split,  query_pos_emb], -1)   #(bs*nhead, len, split_size)
            memory_mean_split_comp = torch.cat([memory_mean_split, mem_pos_emb], -1)     #(bs*nhead, len, split_size)

            strength = 1 + F.relu(self.strength_transform(query_mean_split_comp)) #(bs*nhead, len_query, 1)
            # atten (batch_size, query_size, mem_slot)
            time_attention = temperal_atten(query_mean_split_comp, memory_mean_split_comp, strength)

            #(b, len_mem, outChans, row, col)
            memory_split = torch.split(memory, split_size = self.conv_split_size, dim = 2)
            #(b*len_mem, nhead, conv_split_size, row, col)
            memory_split = torch.stack(memory_split,  1)
            memory_flat = memory_split.view(batch_size*n_head, self.updated_mem_len, -1)
            
            attned_mem = torch.bmm(time_attention, memory_flat) # (b*nhead, len_que, split_size*row*col)
            attned_mem = attned_mem.view(batch_size, n_head, len_query, self.conv_split_size, row, col)
            attned_mem = torch.transpose(attned_mem,2,1).contiguous() #(b, len_que, n_head, split_size, row, col)
            attned_mem = attned_mem.view(batch_size, self.output_chans, row, col) 

            spatial_atten_mem = self.spatial_atten(querys, attned_mem)
            return spatial_atten_mem
        else:
            return None


def match_tensor(out, refer_shape):
    
    skiprow,skipcol = refer_shape
    row, col = out.size()[2], out.size()[3]
    if skipcol >= col:
        pad_col = skipcol - col        
        left_pad_col  = pad_col // 2
        right_pad_col = pad_col - left_pad_col      
        out = F.pad(out, (left_pad_col, right_pad_col, 0,0), mode='reflect')
    else:
        crop_col = col - skipcol
        left_crop_col  = crop_col // 2
        right_col = left_crop_col + skipcol
        out = out[:,:,:, left_crop_col:right_col]
    
    if skiprow >= row:
        pad_row = skiprow - row
        left_pad_row  = pad_row // 2
        right_pad_row = pad_row - left_pad_row
        out = F.pad(out, (0,0, left_pad_row,right_pad_row), mode='reflect')
    else:
        crop_row = row - skiprow   
        left_crop_row  = crop_row // 2
        
        right_row = left_crop_row + skiprow
        
        out = out[:,:,left_crop_row:right_row, :]
    return out


def down_up(inputs, depth = 2):
    # inputs should be (N, C, H, W)
    #down path
    inputs = to_variable(inputs, requires_grad=False, var=True,volatile=True)
    org_dim = len(inputs.size())
    if org_dim == 2:
        inputs = inputs.unsqueeze(0).unsqueeze(0)
    
    size_pool = []
    this_out = inputs
    for idx in range(depth):
        size_pool.append(this_out.size()[2::])
        this_out = F.max_pool2d(this_out, kernel_size = 2)
           
    for didx in range(depth):
        this_size = size_pool[depth - didx - 1]
        this_out = torch.nn.UpsamplingBilinear2d(size=this_size)(this_out)
    if org_dim == 2:
        this_out = this_out[0,0]
    return this_out

def Activation(nchan=0, activ=True):
    if activ is True:
        return nn.ELU(inplace=True)
    elif activ is 'selu':
        print('using selu')
        return nn.SELU(inplace=True)
    else:
        return nn.PReLU(nchan)

class GeneralLinear(nn.Linear):
    def forward(self, inputs):
        inputs_size = inputs.size()
        last_dim = inputs_size[-1]
        out_size = list(inputs.size())
        out_size[-1] = self.out_features

        flat_input = inputs.view(-1, last_dim)
        flat_output = super(GeneralLinear, self).forward(flat_input)

        return flat_output.view(*out_size)

class Dense(nn.Module):
    def __init__(self,  input_dim, out_dim=None, activ='selu'):
        super(Dense, self).__init__()
        if out_dim is None:
            out_dim = input_dim
        self.act = Activation(out_dim, activ = activ)
        self.linear = GeneralLinear(input_dim, out_dim)
        
    def forward(self, x):
        out = self.act(self.linear(x))
        return out

class DenseStack(nn.Module):
    def __init__(self, input_dim, hid_dim, out_dim, depth=0, activ='selu'):
        super(DenseStack, self).__init__()
        self.__dict__.update(locals())
        layers = [Dense(input_dim, hid_dim, activ=activ)]
        for _ in range(depth):
            layers.append(Dense(hid_dim, activ=activ))

        layers.append(GeneralLinear(hid_dim,out_dim))
        self.transform =  nn.Sequential(*layers)
    
    def forward(self,x):
        return self.transform(x)  

def spp(inputs, size_pool=[4]):
    #inputs should be of B*C*row*col
    # the output should be B*C*feat_dim
    feat_dim = np.sum([n**2 for n in size_pool])
    B, C, _, _ = inputs.size()
    output_list = []
    for ks in size_pool:
        this_out = F.adaptive_avg_pool2d(inputs, ks)
        this_out = this_out.view(B, C, ks**2)
        output_list.append(this_out)
    out_tensor = torch.cat(output_list, dim = 2)
    return out_tensor

class LayerNormal(nn.Module):
    ''' Layer normalization module '''

    def __init__(self, d_hid, eps=1e-9):
        super(LayerNormal, self).__init__()

        self.eps = eps
        self.mu = nn.Parameter(torch.ones(1, d_hid, 1, 1), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(1, d_hid, 1, 1), requires_grad=True)

    def forward(self, z):
        _ndim = len(z.size())

        batch_size = z.size()[0]
        z_2d = z.view(batch_size, -1)

        mu   = torch.mean(z_2d, 1, keepdim=True)
        sigma =  torch.std(z_2d, 1, keepdim=True )
        
        #print(mu.size(), sigma.size(), z.size(),flat_z.size())
        
        if _ndim == 4: 
            mu = mu.unsqueeze(-1).unsqueeze(-1)
            sigma = sigma.unsqueeze(-1).unsqueeze(-1)
            #print(mu.size(), sigma.size(), z.size())
            ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
            ln_out = ln_out * self.mu.expand_as(ln_out) + self.beta.expand_as(ln_out)
        else:
            ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
            mu = self.mu.squeeze(-1).squeeze(-1)
            beta = self.beta.squeeze(-1).squeeze(-1)
            ln_out = ln_out * mu.expand_as(ln_out) + beta.expand_as(ln_out)

        return ln_out

class ConvBN(nn.Module):
    def __init__(self, inChans, outChans, activ=True, dilation=1, 
                 act= None, kernel_size = 3, norm='bn'):
        super(ConvBN, self).__init__()
        
        redu = dilation*(kernel_size - 1)
        p1 = redu//2
        p2 = redu - p1
        
        self.conv = nn.Conv2d(inChans, outChans, kernel_size=kernel_size, padding= (p1, p2), dilation=dilation)
        self.norm = getNormLayer(norm)(outChans)
        self.act  = Activation(outChans, activ = activ) if act is not None else passthrough

    def forward(self, x):
        out = self.norm(self.act(self.conv(x)))
        return out

def _make_nConv(inChans, outChans, depth, activ = True, 
                dilation_list=1, norm = 'bn'):
    layers = []
    if type(dilation_list) is not list:
        dilation_list = [dilation_list]*depth
    if depth >= 1 :
        layers.append(ConvBN(inChans, outChans,activ = activ, dilation=dilation_list[0], norm=norm))
        for idx in range(depth-1):
            layers.append(ConvBN(outChans,outChans,activ = activ, dilation=dilation_list[idx+1], norm=norm))
        return nn.Sequential(*layers)
    else:
        return passthrough

class InputTransition(nn.Module):
    def __init__(self,inputChans, outChans, activ=True, norm='bn'):
        self.outChans = outChans
        self.inputChans = inputChans
        super(InputTransition, self).__init__()

        self.conv = nn.Conv2d(inputChans, outChans, kernel_size=3, padding=1)
        self.norm = getNormLayer(norm)(outChans)
        self.act  = Activation(outChans,activ=activ)

    def forward(self, x):
        out = self.norm(self.act(self.conv(x)))
        return out

class DownTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, dropout=False, 
                 activ=True, norm = 'bn', pooling= False):
        super(DownTransition, self).__init__()
        if pooling:
            self.down_conv = nn.Conv2d(inChans, outChans, kernel_size=3, padding=1, stride=1)
            self.max_pooling = nn.MaxPool2d(kernel_size = 2, stride=2)
        else:
            self.down_conv = nn.Conv2d(inChans, outChans, kernel_size=3, padding=1, stride=2)
            self.max_pooling = passthrough
        
        self.norm  = getNormLayer(norm)(outChans)
        self.act1  = Activation(outChans,activ=activ)
        self.drop  = nn.Dropout2d(dropout) if dropout else passthrough
        self.conv_ops = _make_nConv(outChans, outChans, nConvs, activ=activ, norm=norm)
        self.act2 = Activation(outChans,activ=activ)
        
    def forward(self, x):
        down = self.act1(self.down_conv(x))
        down = self.norm(self.max_pooling(down))
        out = self.drop(down)
        out = self.conv_ops(out)
        out = self.act2(out)
        return out


class UpConcat(nn.Module):
    def __init__(self, inChans, hidChans, outChans, nConvs, catChans=None, 
                 dropout=False,stride=2,activ=True, norm='bn'):
        # remeber inChans is mapped to hidChans, then concate together with skipx, the composed channel = outChans
        super(UpConcat, self).__init__()
        #hidChans = outChans // 2
        self.outChans = outChans
        self.drop1 = nn.Dropout2d(dropout) if dropout else passthrough
        self.drop2 = nn.Dropout2d(dropout) if dropout else passthrough
        self.up_conv = nn.ConvTranspose2d(inChans, hidChans, kernel_size=3, 
                                          padding=1, stride=stride, output_padding=1)
        self.norm = getNormLayer(norm)(hidChans)
        self.act1 = Activation(hidChans, activ= activ)
        self.conv_ops = None if catChans is None else \
                      _make_nConv(catChans, self.outChans, depth=nConvs, activ = activ, norm=norm)
        self.act2 = Activation(outChans, activ= activ)

    def forward(self, x, skipx):
        out = self.drop1(x)
        skipxdo = self.drop2(skipx)
        out = self.norm(self.act1(self.up_conv(out)))
        out = match_tensor(out, skipxdo.size()[2:])
        xcat = torch.cat([out, skipxdo], 1)
        if self.conv_ops is None:
           self.conv_ops = _make_nConv(xcat.size()[1], self.outChans, depth=nConvs, activ = activ)
        out  = self.conv_ops(xcat)
        out  = self.act2(out)
        return out

class UpConv(nn.Module):
    def __init__(self, inChans, outChans, dropout=False, stride = 2,activ= True, norm='bn'):
        super(UpConv, self).__init__()
        #hidChans = outChans // 2
        self.up_conv = nn.ConvTranspose2d(inChans, outChans, kernel_size=3, 
                                          padding=1, stride = stride, output_padding=1)
        self.norm = getNormLayer(norm)(outChans)
        self.drop1 = nn.Dropout2d(dropout) if dropout else passthrough
        self.act1 = Activation(outChans,activ= activ)

    def forward(self, x, dest_size):
        '''
        dest_size should be (row, col)
        '''
        out = self.drop1(x)
        out = self.norm(self.act1(self.up_conv(out)))
        out = match_tensor(out, dest_size)
        return out

class OutputTransition(nn.Module):
    def __init__(self, inChans,outChans=1,hidChans=2,activ= True, norm='bn'):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv2d(inChans, hidChans, kernel_size=5, padding=2)
        self.norm   = getNormLayer(norm)(hidChans)
        self.act1 = Activation(hidChans, activ= activ)
        self.conv2 = nn.Conv2d(hidChans,  outChans, kernel_size=1)
        
    def forward(self, x):
        # convolve 32 down to 2 channels
        out = self.norm(self.act1(self.conv1(x)))
        out = self.conv2(out)
        return out

class ConvGRU_cell(nn.Module):
    """Initialize a basic Conv GRU cell.
    Args:
      shape: int tuple thats the height and width of the hidden states h and c
      filter_size: int thself.up_tr256_12   = UpConv(256, 256, 2)at is the height and width of the filters
      num_features: int thats the num of channels of the states, like hidden_size
    """
    def __init__(self, input_chans, filter_size, output_chans, dropout = None, activ='selu'):
        super(ConvGRU_cell, self).__init__()
        
        self.input_chans=input_chans
        self.filter_size=filter_size
        self.output_chans = output_chans
        #self.batch_size=batch_size
        self.dropout = dropout
        self.padding=(filter_size-1)//2 #in this way the output has the same size
        self.register_buffer('device_id', torch.zeros(1))
        self.conv1_inp = nn.Conv2d(input_chans, 3*output_chans,  kernel_size=filter_size,  padding=self.padding, stride=1)
        self.conv1_out = nn.Conv2d(output_chans, 3*output_chans, kernel_size=filter_size, padding=self.padding, stride=1)
        
        self.act1  = Activation(output_chans, activ= activ)

        #self.norm1 = LayerNormal(3*output_chans)
        #self.norm2 = LayerNormal(3*output_chans)
    
    def get_dropmat(self, batch_size):
        W_dropmat = None
        U_dropmat = None
        if self.dropout is not None and self.training is True:
            droprate = 1- self.dropout
            W_dropmat = to_device(torch.bernoulli( droprate * torch.ones(batch_size, self.input_chans, 1, 1)), self.device_id)
            U_dropmat = to_device(torch.bernoulli( droprate * torch.ones(batch_size, self.output_chans, 1, 1)), self.device_id)

        return [W_dropmat, U_dropmat]

    def forward(self, inputs, h_tm1, spatial_gate, dropmat= None):
        # input (B, T, W, H), hidden_state ()
        if  dropmat is not None:
            droprate = 1-self.dropout
            W_drop, U_drop = dropmat
            if self.training is True and W_drop is not None and U_drop is not None:
               inputs  = inputs  *  W_drop.expand_as(inputs) / droprate
               h_tm1   = h_tm1  *  U_drop.expand_as(h_tm1) / droprate

        input_act  = self.conv1_inp(inputs)
        hidden_act = self.conv1_out(h_tm1) 

        (a_inp_r,a_inp_i,a_inp_n) = torch.split(input_act,  self.output_chans,dim=1)#it should return 3 tensors
        (a_hid_r,a_hid_i,a_hid_n) = torch.split(hidden_act, self.output_chans,dim=1)#it should return 3 tensors
        if spatial_gate:
            mean_r_inp = (a_inp_r + a_hid_r)
            mean_r_inp = torch.mean(torch.mean(mean_r_inp, -1, keepdim =True), -2, keepdim =True)

            mean_i_inp = (a_inp_i + a_hid_i)
            mean_i_inp = torch.mean(torch.mean(mean_i_inp, -1, keepdim =True), -2, keepdim =True)

            #print('mean_r_inp shape is: ', mean_r_inp.size())
            r = torch.sigmoid(mean_r_inp).expand_as(a_inp_r)
            i = torch.sigmoid(mean_i_inp).expand_as(a_hid_i)
            n = self.act1(a_inp_n + r * a_hid_n)
            h_t = (1- i)*n  + i*h_tm1 
            
        else:
            r = torch.sigmoid(a_inp_r + a_hid_r)
            i = torch.sigmoid(a_inp_i + a_hid_i)
            n = self.act1(a_inp_n + r * a_hid_n)
            h_t = (1- i)*n  + i*h_tm1 
        return h_t

    def init_hidden(self,batch_size, rowsize, colsize):
        return torch.zeros(batch_size,self.output_chans,rowsize, colsize)

