import torch
import torch.nn as nn
import copy
from .vision_transformer import vit_base_patch16_224 

# --- GNN 子模块 (TGAM) ---
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x, adj):
        """
        x: [B, N, C] (Nodes features)
        adj: [B, N, N] (Adjacency matrix)
        """
        # 简单的图卷积: D^-1 * A * X * W
        # 这里简化为 A * X * W
        out = torch.matmul(adj, x) 
        out = self.linear(out)
        return out + x # Residual connection

class TGAM(nn.Module):
    """
    Topological Graph Alignment Module
    """
    def __init__(self, feature_dim, num_parts=6):
        super(TGAM, self).__init__()
        self.num_parts = num_parts
        self.gcn = GCNLayer(feature_dim, feature_dim)
        self.bn = nn.BatchNorm1d(feature_dim)
        
    def build_adjacency(self, x):
        # 动态构建图：基于特征相似度 (k-NN) 或者 空间邻接关系
        # 这里使用特征相似度构建动态图
        B, N, C = x.shape
        dist = torch.cdist(x, x) # [B, N, N]
        # 取 top-k 邻居构建邻接矩阵
        k = 3
        _, indices = torch.topk(dist, k, largest=False)
        adj = torch.zeros(B, N, N, device=x.device)
        batch_indices = torch.arange(B, device=x.device).view(-1, 1, 1)
        node_indices = torch.arange(N, device=x.device).view(1, -1, 1)
        adj[batch_indices, node_indices, indices] = 1.0
        
        # 归一化
        row_sum = torch.sum(adj, dim=2, keepdim=True) + 1e-6
        adj = adj / row_sum
        return adj

    def forward(self, x):
        # x shape: [B, L, C], where L is sequence length (patches)
        # 我们将 ViT 的输出 token 分组或者直接用作节点
        # 假设 x 已经是除去 CLS token 后的 patch tokens
        
        # 为了减少计算量，我们对 patch 进行空间池化得到 rigid parts
        # 假设 x [B, 196, 768] -> [B, num_parts, 768]
        B, L, C = x.shape
        # 简单实现：将 patch 分割成 num_parts 份进行平均池化
        ratio = L // self.num_parts
        parts = []
        for i in range(self.num_parts):
            part = x[:, i*ratio : (i+1)*ratio, :].mean(dim=1)
            parts.append(part.unsqueeze(1))
        parts_feat = torch.cat(parts, dim=1) # [B, num_parts, C]
        
        # 构建图并卷积
        adj = self.build_adjacency(parts_feat)
        aligned_feat = self.gcn(parts_feat, adj)
        
        # 将图特征展平或池化用于分类
        final_feat = aligned_feat.mean(dim=1) # [B, C]
        return final_feat

# --- 主网络 FD-GNN ---
class FD_GNN(nn.Module):
    def __init__(self, num_classes, camera_num=0, view_num=0):
        super(FD_GNN, self).__init__()
        
        # 1. 基础 Backbone (ViT-Base)
        # 使用 ImageNet 预训练权重
        self.backbone = vit_base_patch16_224(pretrained=True)
        self.in_planes = 768
        
        # 2. FAAM 模块
        self.faam = FAAM(p=0.5, beta=0.1)

        # 3. MDFE (模态解耦)
        # 为了节省显存，我们共享 Backbone 的大部分，只使用独立的 Projection 头
        # Shared Branch
        self.shared_head = nn.Sequential(
            nn.Linear(self.in_planes, self.in_planes),
            nn.BatchNorm1d(self.in_planes),
            nn.ReLU()
        )
        # Specific Branch (轻量级)
        self.specific_head = nn.Sequential(
            nn.Linear(self.in_planes, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        # 4. TGAM (图对齐)
        self.tgam = TGAM(feature_dim=self.in_planes, num_parts=6)

        # 5. Classifier (Bottleneck)
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.classifier = nn.Linear(self.in_planes, num_classes, bias=False)
        self.bottleneck.bias.requires_grad_(False)

    def forward(self, x, label=None, cam_label=None, view_label=None, modal_label=None):
        # x: Input images
        # modal_label: 0 for Optical, 1 for SAR (Assuming batch contains mixed modalities)
        
        # --- Step 1: FAAM ---
        # 这一步需要区分 batch 中的 Optical 和 SAR
        # 实际训练通常是一个 batch 里混合了 O 和 S，为了简单，假设数据加载器已经处理好
        # 或者在这里根据 modal_label 将 batch 拆开再合并
        if self.training and modal_label is not None:
            mask_opt = (modal_label == 0)
            mask_sar = (modal_label == 1)
            if mask_opt.sum() > 0 and mask_sar.sum() > 0:
                x_opt = x[mask_opt]
                x_sar = x[mask_sar]
                x_opt_aug, _ = self.faam(x_opt, x_sar)
                # 将增强后的 Optical 放回去
                x[mask_opt] = x_opt_aug

        # --- Step 2: Backbone Feature Extraction ---
        # ViT 输出 features 通常包含 [CLS] 和 Patch tokens
        # features: [B, 197, 768] (1 CLS + 196 Patches)
        features = self.backbone(x) 
        
        cls_token = features[:, 0]      # [B, 768]
        patch_tokens = features[:, 1:]  # [B, 196, 768]

        # --- Step 3: MDFE (Disentanglement) ---
        feat_shared = self.shared_head(cls_token)   # 共享特征 (用于识别)
        feat_specific = self.specific_head(cls_token) # 模态特定特征 (用于正交约束)

        # --- Step 4: TGAM (Graph Alignment on Patches) ---
        # 使用共享特征空间下的 patch tokens 进行图对齐
        feat_graph = self.tgam(patch_tokens) # [B, 768]

        # --- Step 5: Final Representation ---
        # 融合 CLS 特征和 图特征 (Element-wise sum or Concatenation)
        feat_final = feat_shared + feat_graph

        # Inference 阶段直接返回特征
        if not self.training:
            return feat_final

        # Training 阶段返回 Logits 和 Features 用于计算 Loss
        feat_bn = self.bottleneck(feat_final)
        cls_score = self.classifier(feat_bn)

        return cls_score, feat_final, feat_shared, feat_specific