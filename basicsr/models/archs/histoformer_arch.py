## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange

#########################################################################

Conv2d = nn.Conv2d
##########################################################################
## Layer Norm
def to_2d(x):
    return rearrange(x, 'b c h w -> b (h w c)')

def to_3d(x):
#    return rearrange(x, 'b c h w -> b c (h w)')
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
#    return rearrange(x, 'b c (h w) -> b c h w',h=h,w=w)
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

#        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) #* self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

#        self.weight = nn.Parameter(torch.ones(normalized_shape))
#        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) #* self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type="WithBias"):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Dual-scale Gated Feed-Forward Network (DGFF)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

#        self.dwconv = Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        self.dwconv_5 = Conv2d(hidden_features//4, hidden_features//4, kernel_size=5, stride=1, padding=2, groups=hidden_features//4, bias=bias)
        self.dwconv_dilated2_1 = Conv2d(hidden_features//4, hidden_features//4, kernel_size=3, stride=1, padding=2, groups=hidden_features//4, bias=bias, dilation=2)
#        self.dwconv_dilated2_2 = Conv2d(hidden_features//2, hidden_features//2, kernel_size=3, stride=1, padding=2, groups=hidden_features//2, bias=bias, dilation=2)
#        self.dwconv_dilated3_1 = Conv2d(hidden_features//2, hidden_features//2, kernel_size=5, stride=1, padding=4, groups=hidden_features//2, bias=bias, dilation=2)
#        self.dwconv_dilated3_2 = Conv2d(hidden_features//2, hidden_features//2, kernel_size=3, stride=1, padding=3, groups=hidden_features//2, bias=bias, dilation=3)
#        self.pconv = Conv2d(hidden_features, hidden_features, kernel_size=1, bias=bias)
#        self.pool = nn.AvgPool2d( 3, stride=1, padding=1 , ceil_mode=False , count_include_pad=True , divisor_override=None )
        self.p_unshuffle = nn.PixelUnshuffle(2)
        self.p_shuffle = nn.PixelShuffle(2)

        self.project_out = Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.p_shuffle(x)

        x1, x2 = x.chunk(2, dim=1)
#        x2_1, x2_2 = x2.chunk(2, dim=1)
        x1 = self.dwconv_5(x1)
        x2 = self.dwconv_dilated2_1( x2 )
#        x2_2 = self.dwconv_dilated3_1( x2_2 )
#        x2 = torch.cat([x2_1, x2_2], dim=1)
        x = F.mish( x2 ) * x1
        x = self.p_unshuffle(x)
        x = self.project_out(x)

        
#        x1 = self.dwconv_5(x)
#        x2 = self.dwconv_dilated_2(x)
#        x = F.mish(x2) * x1 + x 

        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.largest_factor = {}

    def softmax_1(self, x, dim=-1):
        logit = x.exp()
        logit  = logit / (logit.sum(dim, keepdim=True) + 1)
        return logit

    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
#        attn = attn.softmax(dim=-1)
        attn = self.softmax_1(attn, dim=-1)
        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class Attention_local(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention_local, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.largest_factor = {}
    
    def find_lagest_factor(self, l):
        if l in self.largest_factor:
            return self.largest_factor[l]
        pass
    def pad(self, x, factor, ds):
        t_pad = []
        for d in ds[::-1]:
            if x.shape[d] % factor != 0:
                t_pad.append(0)
                t_pad.append((x.shape[d]//factor+1)*factor - x.shape[d])
            else:
                t_pad = t_pad + [0,0]
        x = F.pad(x, t_pad, 'constant', 0)
        return x, t_pad
    def unpad(self, x, t_pad):
        _, _, h, w = x.shape
        if len(t_pad) == 0:
            return x
        elif len(t_pad) == 2:            
            return x[:,:,:,t_pad[0]:h-t_pad[1]]
        elif len(t_pad) == 4:
            return x[:,:,t_pad[2]:h-t_pad[3],t_pad[0]:w-t_pad[1]]

    def softmax_1(self, x, dim=-1):
        logit = x.exp()
        logit  = logit / (logit.sum(dim, keepdim=True) + 1)
        return logit
    
    def forward(self, x):
        factor = 4
        b,c,_,_ = x.shape
        qkv1 = self.qkv_dwconv(self.qkv(x))
        q1,k1,v1 = qkv1.chunk(3, dim=1)   
        q1, t_pad = self.pad(q1, factor, [2, 3])
        k1, t_pad = self.pad(k1, factor, [2, 3])
        v1, t_pad = self.pad(v1, factor, [2, 3])
        _,_,h,w = q1.shape
        h1, w1 = h//factor, w//factor
        q1 = rearrange(q1, 'b (head c) (h1 factory1) (w1 factorx1) -> b head (c factorx1 factory1) (h1 w1)', head=self.num_heads, factorx1=factor, factory1=factor)
        k1 = rearrange(k1, 'b (head c) (h1 factory1) (w1 factorx1) -> b head (c factorx1 factory1) (h1 w1)', head=self.num_heads, factorx1=factor, factory1=factor)
        v1 = rearrange(v1, 'b (head c) (h1 factory1) (w1 factorx1) -> b head (c factorx1 factory1) (h1 w1)', head=self.num_heads, factorx1=factor, factory1=factor)

        q1 = torch.nn.functional.normalize(q1, dim=-1)
        k1 = torch.nn.functional.normalize(k1, dim=-1)
        attn1 = (q1 @ k1.transpose(-2, -1)) * self.temperature
#        attn1 = attn1.softmax(dim=-1)
        attn1 = self.softmax_1(attn1, dim=-1)
        out1 = (attn1 @ v1)
        out1 = rearrange(out1, 'b head (c factorx1 factory1) (h1 w1) -> b (head c) (h1 factory1) (w1 factorx1)', head=self.num_heads, factorx1=factor, factory1=factor, w1=w1, h1=h1)
        out1 = self.unpad(out1, t_pad)
        out = self.project_out(out1)
        return out

class Attention_global(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention_global, self).__init__()
        self.factor = 16
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = Conv2d(dim, dim, kernel_size=1, bias=bias)
    
    def find_lagest_factor(self, l):
        if l in self.largest_factor:
            return self.largest_factor[l]
        pass
    def pad(self, x, factor, ds):
        t_pad = []
        for d in ds[::-1]:
            if x.shape[d] % factor != 0:
                t_pad.append(0)
                t_pad.append((x.shape[d]//factor+1)*factor - x.shape[d])
            else:
                t_pad = t_pad + [0,0]
        x = F.pad(x, t_pad, 'constant', 0)
        return x, t_pad
    def unpad(self, x, t_pad):
        _, _, h, w = x.shape
        if len(t_pad) == 0:
            return x
        elif len(t_pad) == 2:
            return x[:,:,:,t_pad[0]:h-t_pad[1]]
        elif len(t_pad) == 4:
            return x[:,:,t_pad[2]:h-t_pad[3],t_pad[0]:w-t_pad[1]]

    def softmax_1(self, x, dim=-1):
        logit = x.exp()
        logit  = logit / (logit.sum(dim, keepdim=True) + 1)
        return logit

    def forward(self, x):
        b,c,_,_ = x.shape
        x, t_pad = self.pad(x, self.factor, [2, 3])
        qkv1 = self.qkv_dwconv(self.qkv(x))
        q2,k2,v2 = qkv1.chunk(3, dim=1)
 
        q2 = F.pixel_unshuffle(q2, self.factor)
        k2 = F.pixel_unshuffle(k2, self.factor)
        v2 = F.pixel_unshuffle(v2, self.factor)
        _, _, h, w = q2.shape
        q2 = rearrange(q2, 'b (head c factory1 factorx1) h w -> b head (factorx1 factory1) (c h w)', head=self.num_heads, factorx1=self.factor, factory1=self.factor)
        k2 = rearrange(k2, 'b (head c factory1 factorx1) h w -> b head (factorx1 factory1) (c h w)', head=self.num_heads, factorx1=self.factor, factory1=self.factor)
        v2 = rearrange(v2, 'b (head c factory1 factorx1) h w -> b head (factorx1 factory1) (c h w)', head=self.num_heads, factorx1=self.factor, factory1=self.factor)
        q2 = torch.nn.functional.normalize(q2, dim=-1)
        k2 = torch.nn.functional.normalize(k2, dim=-1)
        attn2 = (q2 @ k2.transpose(-2, -1)) * self.temperature
#        attn2 = attn2.softmax(dim=-1)
        attn2 = self.softmax_1(attn2, dim=-1)
        out2 = (attn2 @ v2)
        out2 = rearrange(out2, 'b head (factorx1 factory1) (c h w) -> b (head c factory1 factorx1) h w ', head=self.num_heads, factorx1=self.factor, factory1=self.factor, h=h, w=w)
        out2 = F.pixel_shuffle(out2, self.factor)
        out2 = self.unpad(out2, t_pad)

        out = self.project_out(out2)
        return out

    '''
    def forward(self, x):
        factor = 4
        b,c,_,_ = x.shape
        x = self.pixelunshuffle(x)
        qkv1 = self.qkv_dwconv(self.qkv(x))
        q2,k2,v2 = qkv1.chunk(3, dim=1)
 
        q2, t_pad = self.pad(q2, factor, [2, 3])
        k2, t_pad = self.pad(k2, factor, [2, 3])
        v2, t_pad = self.pad(v2, factor, [2, 3])
        _, _, h, w = q2.shape
        h1, w1 = h//factor, w//factor
        q2 = rearrange(q2, 'b (head c) (h1 factory1) (w1 factorx1) -> b head (h1 w1) (c factorx1 factory1)', head=self.num_heads, factorx1=factor, factory1=factor)
        k2 = rearrange(k2, 'b (head c) (h1 factory1) (w1 factorx1) -> b head (h1 w1) (c factorx1 factory1)', head=self.num_heads, factorx1=factor, factory1=factor)
        v2 = rearrange(v2, 'b (head c) (h1 factory1) (w1 factorx1) -> b head (h1 w1) (c factorx1 factory1)', head=self.num_heads, factorx1=factor, factory1=factor)
        q2 = torch.nn.functional.normalize(q2, dim=-1)
        k2 = torch.nn.functional.normalize(k2, dim=-1)
        attn2 = (q2 @ k2.transpose(-2, -1)) * self.temperature
#        attn2 = attn2.softmax(dim=-1)
        attn2 = self.softmax_1(attn2, dim=-1)
        out2 = (attn2 @ v2)
        out2 = rearrange(out2, 'b head (h1 w1) (c factorx1 factory1) -> b (head c) (h1 factory1) (w1 factorx1) ', head=self.num_heads, factorx1=factor, factory1=factor, w1=w1, h1=h1)
        out2 = self.unpad(out2, t_pad)

        out = self.project_out(out2)
        return out
    '''

class Attention_histogram(nn.Module):
    def __init__(self, dim, num_heads, bias, ifBox=True):
        super(Attention_histogram, self).__init__()
        self.factor = num_heads
        self.ifBox = ifBox
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = Conv2d(dim, dim*5, kernel_size=1, bias=bias)
        self.qkv_dwconv = Conv2d(dim*5, dim*5, kernel_size=3, stride=1, padding=1, groups=dim*5, bias=bias)
        self.project_out = Conv2d(dim, dim, kernel_size=1, bias=bias)


    def pad(self, x, factor):
        hw = x.shape[-1]
        t_pad = [0, 0] if hw % factor == 0 else [0, (hw//factor+1)*factor-hw]
        x = F.pad(x, t_pad, 'constant', 0)
        return x, t_pad
    def unpad(self, x, t_pad):
        _, _, hw = x.shape
        return x[:,:,t_pad[0]:hw-t_pad[1]]

    def softmax_1(self, x, dim=-1):
        logit = x.exp()
        logit  = logit / (logit.sum(dim, keepdim=True) + 1)
        return logit

    def normalize(self, x):
        mu = x.mean(-2, keepdim=True)
        sigma = x.var(-2, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) #* self.weight + self.bias
    '''
    def forward(self, x):
        b,c,h,w = x.shape
        x, idx = x.view(b,-1).sort(dim=-1)
        hw = x.shape[-1] // self.factor
        shape_ori = "b (factor hw)" if self.ifBox else "b (hw factor)"
        x = rearrange(x, '{} -> b factor hw 1'.format(shape_ori), factor=self.factor, hw=hw)

        qkv1 = self.qkv(x)
        q2,k2,v2 = qkv1.chunk(3, dim=1) # b,f,hw,1

        shape_ori2 = "b (head c) hw 1"
        q2 = rearrange(q2, '{} -> b head c hw'.format(shape_ori2), head=self.num_heads, hw=hw)
        k2 = rearrange(k2, '{} -> b head c hw'.format(shape_ori2), head=self.num_heads, hw=hw)
        v2 = rearrange(v2, '{} -> b head c hw'.format(shape_ori2), head=self.num_heads, hw=hw)
        q2 = torch.nn.functional.normalize(q2, dim=-1)
        k2 = torch.nn.functional.normalize(k2, dim=-1)
        attn2 = (q2 @ k2.transpose(-2, -1)) * self.temperature
        attn2 = self.softmax_1(attn2, dim=-1)
        out2 = (attn2 @ v2)
        out2 = rearrange(out2, 'b head c hw -> {}'.format(shape_ori2), head=self.num_heads, hw=hw)
        out2 = out2.view(b,-1)
        out2 = torch.scatter(out2, 1, idx, out2).view(b,c,h,w)

        out = self.project_out(out2)
        return out
    '''

    def reshape_attn(self, q, k, v, ifBox):
        b, c = q.shape[:2]
        q, t_pad = self.pad(q, self.factor)
        k, t_pad = self.pad(k, self.factor)
        v, t_pad = self.pad(v, self.factor)
        hw = q.shape[-1] // self.factor
        shape_ori = "b (head c) (factor hw)" if ifBox else "b (head c) (hw factor)"
        shape_tar = "b head (c factor) hw"
        q = rearrange(q, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        k = rearrange(k, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        v = rearrange(v, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = self.softmax_1(attn, dim=-1)
        out = (attn @ v)
        out = rearrange(out, '{} -> {}'.format(shape_tar, shape_ori), factor=self.factor, hw=hw, b=b, head=self.num_heads)
        out = self.unpad(out, t_pad)
        return out

    def forward(self, x):
        b,c,h,w = x.shape
        x_sort, idx_h = x[:,:c//2].sort(-2)
        x_sort, idx_w = x_sort.sort(-1)
        x[:,:c//2] = x_sort
        qkv = self.qkv_dwconv(self.qkv(x))
        q1,k1,q2,k2,v = qkv.chunk(5, dim=1) # b,c,x,x

        v, idx = v.view(b,c,-1).sort(dim=-1)
        q1 = torch.gather(q1.view(b,c,-1), dim=2, index=idx)
        k1 = torch.gather(k1.view(b,c,-1), dim=2, index=idx)
        q2 = torch.gather(q2.view(b,c,-1), dim=2, index=idx)
        k2 = torch.gather(k2.view(b,c,-1), dim=2, index=idx)

        out1 = self.reshape_attn(q1, k1, v, True)
        out2 = self.reshape_attn(q2, k2, v, False)
        
        out1 = torch.scatter(out1, 2, idx, out1).view(b,c,h,w)
        out2 = torch.scatter(out2, 2, idx, out2).view(b,c,h,w)
        out = out1 * out2
        out = self.project_out(out)
        out_replace = out[:,:c//2]
        out_replace = torch.scatter(out_replace, -1, idx_w, out_replace)
        out_replace = torch.scatter(out_replace, -2, idx_h, out_replace)
        out[:,:c//2] = out_replace
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

#        self.norm0 = LayerNorm(dim, LayerNorm_type)
#        self.attn = Attention(dim, num_heads, bias)
        self.attn_g = Attention_histogram(dim, num_heads, bias, True)
        self.norm_g = LayerNorm(dim, LayerNorm_type)
#        self.attn_l = Attention_histogram(dim, num_heads, bias, False)
#        self.norm_l = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
#        self.ffn2 = FeedForward(dim, ffn_expansion_factor, bias)

        self.norm_ff1 = LayerNorm(dim, LayerNorm_type)
#        self.norm_ff2 = LayerNorm(dim, LayerNorm_type)



    def forward(self, x):
#        x = x + self.attn(self.norm0(x))
        x = x + self.attn_g(self.norm_g(x))
#        x2 = x + self.attn_l(self.norm_l(x))
        x_out = x + self.ffn(self.norm_ff1(x))
#        x = x_out
#        x = x + self.attn_g(self.norm_g(x))
#        x_out = x + self.ffn2(self.norm_ff2(x))

        return x_out



##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

class SkipPatchEmbed(nn.Module):
    def __init__(self, in_c=3, dim=48, bias=False):
        super(SkipPatchEmbed, self).__init__()

        self.proj = nn.Sequential(
            nn.AvgPool2d( 2, stride=2, padding=0 , ceil_mode=False , count_include_pad=True , divisor_override=None ),
            Conv2d(in_c, dim, kernel_size=1, bias=bias),
            Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        )

    def forward(self, x, ):
        x = self.proj(x)

        return x

##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

##########################################################################
class Histoformer(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
    ):

        super(Histoformer, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        self.skip_patch_embed1 = SkipPatchEmbed(3, 3)
        self.skip_patch_embed2 = SkipPatchEmbed(3, 3)
        self.skip_patch_embed3 = SkipPatchEmbed(3, 3)
        self.reduce_chan_level_1 = Conv2d(int(dim*2**1)+3, int(dim*2**1), kernel_size=1, bias=bias)
        self.reduce_chan_level_2 = Conv2d(int(dim*2**2)+3, int(dim*2**2), kernel_size=1, bias=bias)
        self.reduce_chan_level_3 = Conv2d(int(dim*2**3)+3, int(dim*2**3), kernel_size=1, bias=bias)

        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = Conv2d(dim, int(dim*2**1), kernel_size=1, bias=bias)
        ###########################
            
        self.output = Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img, ):

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1) # c,h,w

        inp_enc_level2 = self.down1_2(out_enc_level1) # 2c, h/2, w/2
        skip_enc_level1 = self.skip_patch_embed1(inp_img)
        inp_enc_level2 = self.reduce_chan_level_1(torch.cat([inp_enc_level2, skip_enc_level1], 1))

        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        skip_enc_level2 = self.skip_patch_embed2(skip_enc_level1)
        inp_enc_level3 = self.reduce_chan_level_2(torch.cat([inp_enc_level3, skip_enc_level2], 1))

        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)        
        skip_enc_level3 = self.skip_patch_embed3(skip_enc_level2)
        inp_enc_level4 = self.reduce_chan_level_3(torch.cat([inp_enc_level4, skip_enc_level3], 1))

        latent = self.latent(inp_enc_level4) 
                        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)

        ###########################

        out_dec_level1 = self.output(out_dec_level1)
        return out_dec_level1 + inp_img
#        if ifPred:
#            return out_dec_level1[:,:-3] + inp_img, out_dec_level1[:,-3:].mean([2,3])
#
#        return out_dec_level1[:,:-3] + inp_img
        

