import torch
import time
from enum import Enum


class PoolType(Enum):
    MAX = 0
    AVG = 1
    MIN = 2


def getPool1d(poolType, kernel_size = 3, stride = 1, padding = 0):
    if poolType == PoolType.MAX:
        return torch.nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding)
    elif poolType == PoolType.AVG:
        return torch.nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=padding)
    elif poolType == PoolType.MIN:
        return torch.nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding)
    else:
        raise Exception("Invalid PoolType")

def getPool2d(poolType, kernel_size = 3, stride = 1, padding = 0):
    if poolType == PoolType.MAX:
        return torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
    elif poolType == PoolType.AVG:
        return torch.nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
    elif poolType == PoolType.MIN:
        return torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
    else:
        raise Exception("Invalid PoolType")
    

class LocalHolder1D(torch.nn.Module):
    """
    优化后的 1D 信号局部 Holder 指数算法, 输入信号的格式为 (B, C, L), 输出信号的格式为 (B, C, L)
    B: batch size
    C: channel
    L: length
    """
    def __init__(self, offset = 3, poolType = PoolType.MAX):
        """
        offset: 局部 Holder 指数的计算窗口个数，窗口大小从 3 开始，依次为 3, 5, 7, ...；若 offset = 3，则计算窗口大小为 3, 5, 7
        poolType: 局部 Holder 指数的测度类型，目前支持 MAX, AVG, MIN
        """
        super().__init__()
        self.offset = offset
        self.poolType = poolType
        self.pools = [getPool1d(poolType, kernel_size=2 * i + 1, stride=1, padding=i) for i in range(1, offset + 1)]

    def forward(self, input_sig):
        assert input_sig.dim() == 3, "input_sig.dim() should be 3"

        device = input_sig.device
        B, C, L = input_sig.shape
        local_window = torch.tensor([2 * i + 1 for i in range(1, self.offset + 1)], dtype=torch.float32).to(device)
        local_sig = [pool(input_sig).unsqueeze(-1) for pool in self.pools]
        local_sig = torch.cat(local_sig, dim=-1).to(device)

        # 计算局部 Holder 指数
        X = torch.log10(local_window / L).unsqueeze(0).unsqueeze(0)  # (1, 1, offset)
        X = torch.cat([X, torch.ones_like(X)], dim=1).unsqueeze(1)   # (1, 1, 2, offset)
        X = X.expand(1, L, 2, self.offset)                           # (1, L, 2, offset)
        Y = torch.log10(local_sig)                                   # (B, C, L, offset)
        
        # (1, L, 2, offset) @ (1, L, offset, 2) = (1, L, 2, 2) -> Inverse
        # (1, L, 2, 2) @ (1, L, 2, offset) = (1, L, 2, offset)
        # (1, L, 2, offset) @ (B, L, offset, C) = (B, L, 2, C)
        coeff = torch.inverse(X @ X.transpose(2, 3)) @ X @ Y.permute(0, 2, 3, 1).to(device)
        coeff = coeff.transpose(2, 3)                                # (B, L, C, 2)
        holderSig = coeff[:, :, :, 0].squeeze(-1).transpose(1, 2)    # (B, C, L)
        
        return holderSig


class LocalHolder2D(torch.nn.Module):
    """
    计算 2D 信号的局部 Holder 指数, 输入信号的格式为 (B, C, H, W), 输出信号的格式为 (B, C, H, W)
    B: batch size
    C: channel
    H: height
    W: width
    """
    def __init__(self, offset = 3, poolType = PoolType.MAX):
        """
        offset: 局部 Holder 指数的计算窗口个数，窗口大小从 3 开始，依次为 3, 5, 7, ...；若 offset = 3，则计算窗口大小为 3, 5, 7
        poolType: 局部 Holder 指数的测度类型，目前支持 MAX, AVG, MIN
        """
        super().__init__()
        self.offset = offset
        self.poolType = poolType
        self.pools = [getPool2d(poolType, kernel_size=2 * i + 1, stride=1, padding=i) for i in range(1, offset + 1)]

    def forward(self, x):
        assert x.dim() == 4, "x.dim() should be 4"

        device = x.device
        B, C, H, W = x.shape
        local_window = torch.tensor([2 * i + 1 for i in range(1, self.offset + 1)], dtype=torch.float32).to(device)
        local_img = [pool(x).unsqueeze(-1) for pool in self.pools]
        local_img = torch.cat(local_img, dim=-1).to(device)

        # 计算局部 Holder 指数
        X = torch.log10(local_window / (H * W)).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        X = torch.cat([X, torch.ones_like(X)], dim=-2).unsqueeze(1)
        X = X.expand(1, H, W, 2, self.offset)
        Y = torch.log10(local_img)

        # (1, H, W, 2, offset) @ (1, H, W, offset, 2) = (1, H, W, 2, 2) -> Inverse
        # (1, H, W, 2, 2) @ (1, H, W, 2, offset) = (1, H, W, 2, offset)
        # (1, H, W, 2, offset) @ (B, H, W, offset, C) = (B, H, W, 2, C)
        coeff = torch.inverse(X @ X.transpose(-2, -1)) @ X @ Y.permute(0, 2, 3, 4, 1).to(device)
        coeff = coeff.transpose(-2, -1)                          # (B, H, W, C, 2)
        holderImg = coeff[:, :, :, :, 0].permute(0, 3, 1, 2)     # (B, C, H, W)

        return holderImg


class SPSD1D(torch.nn.Module):
    """
    计算 1D 信号的 SPSD, 输入信号的格式为 (B, C, L), 输出信号的格式为 (B, C, N)
    B: batch size
    C: channel
    N: number of alpha set
    """
    def __init__(self, minAlpha = 0, maxAlpha = 3, N = 64, offset = 3, poolType = PoolType.MAX):
        """
        minAlpha: Holder 指数的最小值
        maxAlpha: Holder 指数的最大值
        N: Holder 指数的个数
        offset: 局部 Holder 指数的计算窗口个数，窗口大小从 3 开始，依次为 3, 5, 7, ...；若 offset = 3，则计算窗口大小为 3, 5, 7
        poolType: 局部 Holder 指数的测度类型，目前支持 MAX, AVG, MIN
        """
        super().__init__()
        self.minAlpha = minAlpha
        self.maxAlpha = maxAlpha
        self.N = N
        self.delta = (maxAlpha - minAlpha) / N
        self.localholder = LocalHolder1D(offset = offset, poolType = poolType)

    def forward(self, x):
        device = x.device
        holderx = self.localholder(x)
        maxHolder = torch.max(holderx, dim=-1).values.unsqueeze(-1)
        alpha = torch.arange(self.N, dtype=torch.float32, device=device) * self.delta + self.minAlpha
        condition = (holderx >= alpha[:, None, None, None]) & (holderx < (alpha + self.delta)[:, None, None, None])
        
        # size = torch.sum(condition, dim=-1).to(device)
        # size[size == 0] = 1
        # SPSD = torch.sum((x * condition) ** 2, dim=-1) / size

        SPSD = torch.sum((x * condition) ** 2, dim=-1)
        SPSD[torch.isnan(SPSD)] = 0
        return SPSD.permute(1, 2, 0), maxHolder


class SPSD2D(torch.nn.Module):
    """
    计算 2D 信号的 SPSD, 输入信号的格式为 (B, C, H, W), 输出信号的格式为 (B, C, N)
    B: batch size
    C: channel
    N: number of alpha set
    """
    def __init__(self, minAlpha = 0, maxAlpha = 3, N = 64, offset = 3, poolType = PoolType.MAX):
        """
        minAlpha: Holder 指数的最小值
        maxAlpha: Holder 指数的最大值
        N: Holder 指数的个数
        offset: 局部 Holder 指数的计算窗口个数，窗口大小从 3 开始，依次为 3, 5, 7, ...；若 offset = 3，则计算窗口大小为 3, 5, 7
        poolType: 局部 Holder 指数的测度类型，目前支持 MAX, AVG, MIN
        """
        super().__init__()
        self.minAlpha = minAlpha
        self.maxAlpha = maxAlpha
        self.N = N
        self.delta = (maxAlpha - minAlpha) / N
        self.localholder = LocalHolder2D(offset = offset, poolType = poolType)

    def forward(self, x):
        device = x.device
        holderx = self.localholder(x)

        maxHolder = torch.max(holderx, dim=-1).values.to(device)
        maxHolder = torch.max(maxHolder, dim=-1).values

        alpha = torch.arange(self.N, dtype=torch.float32, device=device) * self.delta + self.minAlpha
        condition = (holderx >= alpha[:, None, None, None, None]) & \
                    (holderx < (alpha + self.delta)[:, None, None, None, None])

        # size = torch.sum(condition, dim=(-1, -2)).to(device)
        # size[size == 0] = 1
        # SPSD = torch.sum(torch.sum((x * condition) ** 2, dim=-1), dim=-1) / size

        SPSD = torch.sum(torch.sum((x * condition) ** 2, dim=-1), dim=-1)

        SPSD[torch.isnan(SPSD)] = 0
        return SPSD.permute(1, 2, 0), maxHolder.unsqueeze(-1)
    

class PWVD(torch.nn.Module):
    """
    计算 2D 信号的 PWVD, 输入信号的格式为 (B, C, H, W), 输出信号的格式为 (B, C, H, W, N, N), 其中 N 为 PWVD 的计算区间大小
    B: batch size
    C: channel
    H: height
    W: width
    N: PWVD Interval size
    """
    def __init__(self, N):
        """
        N: PWVD Interval size
        """
        super().__init__()
        self.N = N
        self.paddings = (N - 1) // 2

    def forward(self, x):
        device = x.device
        pad = self.paddings
        B, C, H, W = x.shape

        idx_i = torch.arange(pad, H + pad).view(-1, 1, 1, 1) + torch.arange(self.N).view(1, 1, -1, 1) - pad
        idx_j = torch.arange(pad, W + pad).view(1, -1, 1, 1) + torch.arange(self.N).view(1, 1, 1, -1) - pad
        idx_i, idx_j = idx_i.to(device), idx_j.to(device)
        
        x = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode='constant', value=0)
        x = x[:, :, idx_i, idx_j]
        x = x * torch.conj(x.flip([-2, -1]))  # autocorrelation
        x = torch.abs(torch.fft.fftshift(torch.fft.fftn(x, dim=(-2, -1)), dim=(-2, -1)))

        return x


class SIFT(torch.nn.Module):
    """
    计算 2D 信号的 SIFT, 输入信号的格式为 (B, C, H, W), 输出信号的格式为 (B, C, H, W, N), 其中 N 为统计的 Alpha 区间的个数
    B: batch size
    C: channel
    H: height
    W: width
    N: number of alpha set
    """
    def __init__(self, N1 = 5, minAlpha = 0, maxAlpha = 9, N2 = 9, offset = 3, poolType = PoolType.MAX):
        """
        N1: PWVD 的计算区间大小
        minAlpha: Holder 指数的最小值
        maxAlpha: Holder 指数的最大值
        N2: Holder 指数的个数
        offset: 局部 Holder 指数的计算窗口个数，窗口大小从 3 开始，依次为 3, 5, 7, ...；若 offset = 3, 则计算窗口大小为 3, 5, 7
        poolType: 局部 Holder 指数的测度类型，目前支持 MAX, AVG, MIN
        """
        super().__init__()
        self.minAlpha = minAlpha
        self.maxAlpha = maxAlpha
        self.N1 = N1
        self.N2 = N2
        self.delta = (maxAlpha - minAlpha) / N2
        self.pwvd = PWVD(N1)
        self.sps  = SPSD2D(minAlpha, maxAlpha, N2, offset, poolType)

    def forward(self, x):
        device = x.device
        B, C, H, W = x.shape

        x = self.pwvd(x)
        x = x.reshape(B * C, H * W, self.N1, self.N1)
        x, _ = self.sps(x)

        return x.reshape(B, C, H, W, -1)









