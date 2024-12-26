import torch
import torch.nn as nn
class TConv(nn.Sequential):
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
    ):
        super().__init__(
            Convolution(spatial_dims, in_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1),
            Convolution(spatial_dims, out_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1),
        )
class UpCat(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        cat_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        interp_mode: str = "linear",
        align_corners: Optional[bool] = True,
        halves: bool = True,
    ):
        super().__init__()
        up_chns = in_chns // 2 if halves else in_chns
        self.upsample = UpSample(
            spatial_dims,
            in_chns,
            up_chns,
            2,
            mode=upsample,
            interp_mode=interp_mode,
            align_corners=align_corners,
        )
        self.convs = TConv(spatial_dims, cat_chns + up_chns, out_chns, act, norm, bias, dropout)

    def forward(self, x: torch.Tensor, x_e: Optional[torch.Tensor]):
        x_0 = self.upsample(x)
        if x_e is not None:
            for i in range(len(x.shape) - 2):
                if x_e.shape[-i - 1] != x_0.shape[-i - 1]:
                    x_0 = torch.nn.functional.pad(x_0, [0, 1] * (len(x.shape) - 2), "replicate")
            x = torch.cat([x_e, x_0], dim=1)
        return self.convs(x)


class STEM(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(STEM, self).__init__()
        self.conv = nn.Sequential(
            DOConv2d(ch_in, ch_out, kernel_size=3,  padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class FE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            DOConv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            DOConv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        if out_channels == in_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        h = self.in_layers(x)
        h = self.out_layers(h)
        return h + self.skip(x)
import math
import torch
import numpy as np
from torch.nn import init
from itertools import repeat
from torch.nn import functional as F
from torch._jit_internal import Optional
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from BAM import BAM
import collections
class DOConv2d(Module):

    __constants__ = ['stride', 'padding', 'dilation', 'groups',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size', 'D_mul']
    __annotations__ = {'bias': Optional[torch.Tensor]}

    def __init__(self, in_channels, out_channels, kernel_size, D_mul=None, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(DOConv2d, self).__init__()

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
                valid_padding_modes, padding_mode))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self._padding_repeated_twice = tuple(x for x in self.padding for _ in range(2))
        M = self.kernel_size[0]
        N = self.kernel_size[1]
        self.D_mul = M * N if D_mul is None or M * N <= 1 else D_mul
        self.W = Parameter(torch.Tensor(out_channels, in_channels // groups, self.D_mul))
        init.kaiming_uniform_(self.W, a=math.sqrt(5))

        if M * N > 1:
            self.D = Parameter(torch.Tensor(in_channels, M * N, self.D_mul))
            init_zero = np.zeros([in_channels, M * N, self.D_mul], dtype=np.float32)
            self.D.data = torch.from_numpy(init_zero)

            eye = torch.reshape(torch.eye(M * N, dtype=torch.float32), (1, M * N, M * N))
            d_diag = eye.repeat((in_channels, 1, self.D_mul // (M * N)))
            if self.D_mul % (M * N) != 0:  # the cases when D_mul > M * N
                zeros = torch.zeros([in_channels, M * N, self.D_mul % (M * N)])
                self.d_diag = Parameter(torch.cat([d_diag, zeros], dim=2), requires_grad=False)
            else:  # the case when D_mul = M * N
                self.d_diag = Parameter(d_diag, requires_grad=False)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.W)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(DOConv2d, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'

    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        M = self.kernel_size[0]
        N = self.kernel_size[1]
        DoW_shape = (self.out_channels, self.in_channels // self.groups, M, N)
        if M * N > 1:
            D = self.D + self.d_diag
            W = torch.reshape(self.W, (self.out_channels // self.groups, self.in_channels, self.D_mul))

            DoW = torch.reshape(torch.einsum('ims,ois->oim', D, W), DoW_shape)

        else:

            DoW = torch.reshape(self.W, DoW_shape)
        return self._conv_forward(input, DoW)


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


_pair = _ntuple(2)

class FeatureEncoder(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 2,
        in_channels: int = 1,
        out_channels: int = 2,
        features: Sequence[int] = (64, 128, 256, 512, 1024, 128),
        act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: Union[str, tuple] = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: Union[float, tuple] = 0.0,
        dimensions: Optional[int] = None,
    ):

        super().__init__()
        if dimensions is not None:
            spatial_dims = dimensions

        fea = ensure_tuple_rep(features, 6)
        print(f"BasicUNet features: {fea}.")

        self.fe1 = FE(64, 64)
        self.fe2 = FE(128, 128)
        self.fe3 = FE(256, 256)
        self.fe4 = FE(512, 512)

        self.bam1 = BAM(128)
        self.bam2 = BAM(256)
        self.bam3 = BAM(512)
        self.bam4 = BAM(1024)

        self.conv_0 = TConv(2, 3, features[0], act, norm, bias, dropout)
        self.down_1 = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout)
        self.down_2 = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout)
        self.down_3 = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout)
        self.down_4 = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout)


        self.stem = STEM(3,features[0])

    def forward(self, x: torch.Tensor):
        x0 = self.stem(x)
        x0 = self.fe1(x0)

        x1 = self.down_1(x0)

        x1 = self.fe2(x1)
        x2 = self.down_2(x1)

        x2 = self.fe3(x2)
        x3 = self.down_3(x2)

        x3 = self.fe4(x3)
        x4 = self.down_4(x3)

        x1 = self.bam1(x1)
        x2 = self.bam2(x2)
        x3 = self.bam3(x3)
        x4 = self.bam4(x4)

        return [x0, x1, x2, x3, x4]
import torch
import torch.nn as nn
import torch.nn.functional as F

class DCGM(nn.Module):
    def __init__(self, dim, heads=4):
        super(DCGM, self).__init__()
        self.heads = heads
        self.dim = dim

        # Depthwise convolution
        self.dwconv = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, groups=dim // 2)

        # QKV projection layer
        self.qkvl = nn.Conv2d(dim // 2, (dim // 4) * heads, kernel_size=1)

        # Pooling layers
        self.pool_q = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.pool_k = nn.MaxPool2d(kernel_size=2, stride=2)

        # Activation functions
        self.act = nn.GELU()
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()

        # Convolution layers
        self.conv3x3 = nn.Conv2d((dim // 4) * heads, dim, kernel_size=3, padding=1)
        self.conv1x1 = nn.Conv2d(dim, dim // 2, kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(dim // 4, dim // 2, kernel_size=1)
        self.conv1x1_3 = nn.Conv2d(dim // 2, dim, kernel_size=1)

        # Depthwise convolution with batch normalization
        self.dwc3x3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.bn = nn.BatchNorm2d(dim)

    def channel_shuffle(self, x, groups=2):
        """Performs channel shuffle operation."""
        B, C, H, W = x.shape
        assert C % groups == 0, "Channels must be divisible by groups"

        x = x.reshape(B, groups, C // groups, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return x.reshape(B, C, H, W)

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.dim, f"Input channel size {C} must match model dim {self.dim}"

        # Split input channels into two parts
        x1, x2 = torch.split(x, C // 2, dim=1)

        # Process x2
        x2 = self.act(self.qkvl(x2))
        x2 = self.relu(self.conv3x3(x2))
        x2 = self.conv1x1(x2)

        # Process x1
        x1 = self.act(self.qkvl(x1)).reshape(B, self.heads, C // self.heads, H, W)
        q = torch.sum(x1[:, :-3], dim=1)
        k = x1[:, -3]
        v = x1[:, -2].flatten(2)

        # Pooling for Q and K
        q = self.pool_q(q)
        k = self.pool_k(k)

        # Attention mechanism
        qk = torch.softmax(torch.matmul(q.flatten(2), k.flatten(2).transpose(1, 2)), dim=1).transpose(1, 2)
        x1 = torch.matmul(qk, v).reshape(B, C // 4, H, W)
        x1 = self.conv1x1_2(x1)

        # Concatenate x1 and x2
        x = torch.cat([x1, x2], dim=1)
        x = self.channel_shuffle(x, groups=2)

        # Depthwise convolution and batch normalization
        x = self.gelu(self.bn(self.dwc3x3(x)))

        # Ensure shapes match before addition
        if x1.shape != x.shape:
            x1 = F.interpolate(x1, size=x.shape[2:], mode="bilinear", align_corners=False)

        x1 = self.conv1x1_3(x1)
        return x + x1
class UNetDe(nn.Module):
    @deprecated_arg(
        name="dimensions", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead."
    )
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        features: Sequence[int] = (32, 32, 64, 128, 256, 32),
        act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: Union[str, tuple] = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        dimensions: Optional[int] = None,
    ):

        super().__init__()
        if dimensions is not None:
            spatial_dims = dimensions

        fea = ensure_tuple_rep(features, 6)
        print(f"BasicUNet features: {fea}.")

        self.dcgm = DCGM(dim=1024)

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(128,
                            512),
            torch.nn.Linear(512,
                            512),
        ])
        self.conv_0 = TConv(spatial_dims, in_channels, features[0], act, norm, bias, dropout)
        self.down_1 = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout)
        self.down_2 = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout)
        self.down_3 = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout)
        self.down_4 = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout)

        self.upcat_4 = UpCat(spatial_dims, fea[4], fea[3], fea[3], act, norm, bias, dropout, upsample)

        self.upcat_3 = UpCat(spatial_dims, fea[3], fea[2], fea[2], act, norm, bias, dropout, upsample)

        self.upcat_2 = UpCat(spatial_dims, fea[2], fea[1], fea[1], act, norm, bias, dropout, upsample)

        self.upcat_1 = UpCat(spatial_dims, fea[1], fea[0], fea[5], act, norm, bias, dropout, upsample, halves=False)

        self.PH = nn.Sequential(
            FE(128, 64), FE(64, 64), nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self,  x: torch.Tensor, t, embeddings=None, image=None):
        temb = get_timestep_embedding(t, 128)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)
        outputs = []

        if image is not None :

            x = torch.cat([image, x], dim=1)

        # print("cat.shape",x.shape)
        # print("temb.shape",temb.shape)
        x0 = self.conv_0(x, temb)

        # print("x0.shape", x0.shape)
        if embeddings is not None:
            x0 += embeddings[0]
            # print("emb+x0.shape", x0.shape)

        x1 = self.down_1(x0, temb)

        # print("x1.shape", x1.shape)
        if embeddings is not None:
            x1 += embeddings[1]
            # print("emb+x1.shape", x1.shape)
        x2 = self.down_2(x1, temb)

        # print("x2.shape", x2.shape)
        if embeddings is not None:
            x2 += embeddings[2]
            # print("emb+x2.shape", x2.shape)
        x3 = self.down_3(x2, temb)
        # print("x3.shape", x3.shape)
        if embeddings is not None:
            x3 += embeddings[3]
            # print("emb+x3.shape", x3.shape)
        x4 = self.down_4(x3, temb)

        x4 = self.dcgm(x4)
        # print("x4.shape", x4.shape)
        if embeddings is not None:
            x4 += embeddings[4]
            # print("emb+x4.shape", x4.shape)

        u4 = self.upcat_4(x4, x3, temb)

        u3 = self.upcat_3(u4, x2, temb)

        u2 = self.upcat_2(u3, x1, temb)

        u1 = self.upcat_1(u2, x0, temb)

        out = self.PH(u1)

        return  out