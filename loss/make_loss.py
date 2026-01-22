# encoding: utf-8

import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss


class OrthogonalityLoss(nn.Module):
    def __init__(self):
        super(OrthogonalityLoss, self).__init__()

    def forward(self, f_shared, f_specific):
        # 使得共享特征和特定特征正交（点积趋近于0）
        # f_shared: [B, C], f_specific: [B, C_small]
        # 先归一化
        f_shared_norm = F.normalize(f_shared, p=2, dim=1)
        f_specific_norm = F.normalize(f_specific, p=2, dim=1)
        
        # 计算余弦相似度矩阵
        # 如果维度不同，需要投影，这里假设我们在 model 里已经处理好或者只约束共有维度
        # 简单起见，我们在 model 里把 specific 投影回同维度，或者只计算 batch 内的相关性
        
        # 简化版正交损失：最小化相互相关性
        # 这里用简单的点积平方和
        # 如果维度不一样，无法直接点积。建议在 Loss 前加个线性层对齐，或者只最小化 Softmax 分布差异(KL散度最大化)。
        
        # 按照论文标准做法：
        # Loss = || f_shared^T * f_specific ||_F^2
        # 需要维度一致，若不一致建议在 Model 输出前用 Linear 对齐
        return torch.mean(torch.abs(torch.mm(f_shared_norm, f_specific_norm.t())))

def make_loss(cfg, num_classes):
    # ... 原有的 Loss 定义 ...
    triplet = TripletLoss(margin=cfg.SOLVER.MARGIN)
    xent = CrossEntropyLabelSmooth(num_classes=num_classes)
    orth_loss_func = OrthogonalityLoss()

    def loss_func(score, feat, feat_shared, feat_specific, target):
        # 1. ID Loss
        loss_id = xent(score, target)
        
        # 2. Triplet Loss (基于最终融合特征)
        loss_tri = triplet(feat, target)[0]
        
        # 3. Orthogonality Loss (新加的)
        # 确保 feat_shared 和 feat_specific 维度兼容，或调整实现
        loss_orth = orth_loss_func(feat_shared, feat_specific) * 0.1 # 权重 alpha
        
        # 总损失
        return loss_id + loss_tri + loss_orth
        
    return loss_func


