"""
Pytorch implementation of U-Net based on Kolmogorov-Arnold Network, based on the U-Net implementation in https://github.com/milesial/Pytorch-UNet.
The U-Net model is modified to use the FastKANConvLayer instead of the Conv2d layer in the original implementation.
The Convolution operation is implemented in https://github.com/XiangboGaoBarry/KA-Conv
"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple, Union

class PolynomialFunction(nn.Module):
    def __init__(self, degree: int = 3):
        super().__init__()
        self.degree = degree

    def forward(self, x):
        return torch.stack([x ** i for i in range(self.degree)], dim=-1)
    
class BSplineFunction(nn.Module):
    def __init__(self, grid_min: float = -2., grid_max: float = 2., degree: int = 3, num_basis: int = 8):
        super().__init__()
        self.degree = degree
        self.num_basis = num_basis
        self.knots = torch.linspace(grid_min, grid_max, num_basis + degree + 1)  # Uniform knots

    def basis_function(self, i, k, t):
        if k == 0:
            return ((self.knots[i] <= t) & (t < self.knots[i + 1])).float()
        else:
            left_num = (t - self.knots[i]) * self.basis_function(i, k - 1, t)
            left_den = self.knots[i + k] - self.knots[i]
            left = left_num / left_den if left_den != 0 else 0

            right_num = (self.knots[i + k + 1] - t) * self.basis_function(i + 1, k - 1, t)
            right_den = self.knots[i + k + 1] - self.knots[i + 1]
            right = right_num / right_den if right_den != 0 else 0

            return left + right 
    
    def forward(self, x):
        x = x.squeeze()  # Assuming x is of shape (B, 1)
        basis_functions = torch.stack([self.basis_function(i, self.degree, x) for i in range(self.num_basis)], dim=-1)
        return basis_functions

class ChebyshevFunction(nn.Module):
    def __init__(self, degree: int = 4):
        super().__init__()
        self.degree = degree

    def forward(self, x):
        chebyshev_polynomials = [torch.ones_like(x), x]
        for n in range(2, self.degree):
            chebyshev_polynomials.append(2 * x * chebyshev_polynomials[-1] - chebyshev_polynomials[-2])
        return torch.stack(chebyshev_polynomials, dim=-1)

class FourierBasisFunction(nn.Module):
    def __init__(self, num_frequencies: int = 4, period: float = 1.0):
        super().__init__()
        assert num_frequencies % 2 == 0, "num_frequencies must be even"
        self.num_frequencies = num_frequencies
        self.period = nn.Parameter(torch.Tensor([period]), requires_grad=False)

    def forward(self, x):
        frequencies = torch.arange(1, self.num_frequencies // 2 + 1, device=x.device)
        sin_components = torch.sin(2 * torch.pi * frequencies * x[..., None] / self.period)
        cos_components = torch.cos(2 * torch.pi * frequencies * x[..., None] / self.period)
        basis_functions = torch.cat([sin_components, cos_components], dim=-1)
        return basis_functions
        
class RadialBasisFunction(nn.Module):
    def __init__(self, grid_min: float = -2., grid_max: float = 2., num_grids: int = 4, denominator: float = None):
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    def forward(self, x):
        return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)
    

class SplineConv1D(nn.Conv1d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int]] = 3,
                 stride: Union[int, Tuple[int]] = 1, padding: Union[int, Tuple[int]] = 0, 
                 dilation: Union[int, Tuple[int]] = 1, groups: int = 1, bias: bool = True, 
                 init_scale: float = 0.1, padding_mode: str = "zeros", **kw) -> None:
        self.init_scale = init_scale
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, **kw)

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class FastKANConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int]] = 3,
                 stride: Union[int, Tuple[int]] = 1, padding: Union[int, Tuple[int]] = 0, 
                 dilation: Union[int, Tuple[int]] = 1, groups: int = 1, bias: bool = True, 
                 grid_min: float = -2., grid_max: float = 2., num_grids: int = 4, use_base_update: bool = True, 
                 base_activation = F.silu, spline_weight_init_scale: float = 0.1, padding_mode: str = "zeros",
                 kan_type: str = "RBF") -> None:
        
        super().__init__()
        self.rbf = self._get_basis_function(kan_type, grid_min, grid_max, num_grids)
        self.spline_conv = SplineConv1D(in_channels * num_grids, out_channels, kernel_size, stride, padding, dilation, groups, bias, spline_weight_init_scale, padding_mode)
        
        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation
            self.base_conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)

    def _get_basis_function(self, kan_type, grid_min, grid_max, num_grids):
        if kan_type == "RBF":
            return RadialBasisFunction(grid_min, grid_max, num_grids)
        elif kan_type == "Fourier":
            return FourierBasisFunction(num_grids)
        elif kan_type == "Poly":
            return PolynomialFunction(num_grids)
        elif kan_type == "Chebyshev":
            return ChebyshevFunction(num_grids)
        elif kan_type == "BSpline":
            return BSplineFunction(grid_min, grid_max, 3, num_grids)

    def forward(self, x):
        batch_size, channels, length = x.shape
        x_rbf = self.rbf(x.view(batch_size, channels, -1)).view(batch_size, channels, length, -1)
        x_rbf = x_rbf.permute(0, 3, 1, 2).contiguous().view(batch_size, -1, length)
        
        # Apply spline convolution
        ret = self.spline_conv(x_rbf)
        
        if self.use_base_update:
            base = self.base_conv(self.base_activation(x))
            ret = ret + base
        
        return ret


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, device):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device

        self.double_conv = nn.Sequential(
            FastKANConvLayer(self.in_channels, self.out_channels//2, padding=1, kernel_size=3, stride=1, kan_type='RBF'),
            nn.BatchNorm1d(self.out_channels//2),
            nn.ReLU(inplace=True),
            FastKANConvLayer(self.out_channels//2, self.out_channels, padding=1, kernel_size=3, stride=1, kan_type='RBF'),
            nn.BatchNorm1d(self.out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, device='mps'):
        super().__init__()
        self.device = device
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_channels, out_channels, device=self.device)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, device='mps'):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, device=device)
        else:
            self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CL
        diffL = x2.size()[2] - x1.size()[2]

        x1 = F.pad(x1, [diffL // 2, diffL - diffL // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = FastKANConvLayer(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class KANU_Net(nn.Module):
    def __init__(self, n_channels, n_out, bilinear=True, device='mps'):
        super().__init__()
        self.n_channels = n_channels
        self.n_out = n_out
        self.bilinear = bilinear
        self.device = device

        self.channels = [64, 128, 256, 512, 1024]

        self.inc = DoubleConv(n_channels, 64, device=self.device)
        self.down1 = Down(self.channels[0], self.channels[1], self.device)
        self.down2 = Down(self.channels[1], self.channels[2], self.device)
        self.down3 = Down(self.channels[2], self.channels[3], self.device)
        factor = 2 if bilinear else 1
        self.down4 = Down(self.channels[3], self.channels[4] // factor, self.device)
        self.up1 = Up(self.channels[4], self.channels[3] // factor, bilinear, self.device)
        self.up2 = Up(self.channels[3], self.channels[2] // factor, bilinear, self.device)
        self.up3 = Up(self.channels[2], self.channels[1] // factor, bilinear, self.device)
        self.up4 = Up(self.channels[1], self.channels[0], bilinear, self.device)
        self.outc = OutConv(self.channels[0], n_out)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

# from torchinfo import summary

# device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
# print(device)
# model = KANU_Net(1, 1, 'mps').to(device)
# x = torch.randn((2, 1, 224)).to(device).requires_grad_(True)  # 1D input
# summary(model, input_data=x)

# y = model(x)
# print(y.shape)

# dx = torch.autograd.grad(y, x, torch.ones_like(y), retain_graph=True)[0]
# print(dx)

# dxx = torch.autograd.grad(dx, x, torch.ones_like(dx), retain_graph=True)[0]
# print(dxx)

