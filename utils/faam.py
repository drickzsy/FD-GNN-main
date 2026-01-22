#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.fft

class FAAM(nn.Module):
    """
    Frequency-Adaptive Augmentation Module (FAAM)
    将 SAR 的幅度谱（风格）混合到 Optical 的幅度谱中，保持 Optical 的相位谱（结构）。
    """
    def __init__(self, p=0.5, beta=0.1):
        super(FAAM, self).__init__()
        self.p = p          # 执行概率
        self.beta = beta    # 混合比例

    def extract_amp_phase(self, fft_img):
        # 获取幅度谱和相位谱
        amp = torch.abs(fft_img)
        phase = torch.angle(fft_img)
        return amp, phase

    def forward(self, x_opt, x_sar):
        """
        x_opt: Optical images [B, C, H, W]
        x_sar: SAR images [B, C, H, W]
        """
        if not self.training or torch.rand(1) > self.p:
            return x_opt, x_sar

        # 1. 快速傅里叶变换
        fft_opt = torch.fft.fft2(x_opt, dim=(-2, -1))
        fft_sar = torch.fft.fft2(x_sar, dim=(-2, -1))

        # 2. 提取幅度与相位
        amp_opt, pha_opt = self.extract_amp_phase(fft_opt)
        amp_sar, pha_sar = self.extract_amp_phase(fft_sar)

        # 3. 幅度谱混合 (低频中心化操作通常在fftshift后做，这里简化为直接混合)
        # 很多频域增强为了简化，只交换中心区域(低频)，这里演示全局混合+掩码的逻辑
        # 定义一个简单的掩码 M (控制混合区域，这里简化为全局线性插值)
        
        amp_opt_aug = (1 - self.beta) * amp_opt + self.beta * amp_sar

        # 4. 逆变换重构图像 (使用 Optical 的相位 + 混合后的幅度)
        fft_aug = amp_opt_aug * torch.exp(1j * pha_opt)
        x_opt_aug = torch.fft.ifft2(fft_aug, dim=(-2, -1)).real

        # 保持数值稳定性
        x_opt_aug = torch.clamp(x_opt_aug, 0, 1)

        return x_opt_aug, x_sar