import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
from loss import clip_loss


def do_train_pair(cfg, model, train_loader_pair, val_loader, optimizer, scheduler, num_query, local_rank):
    # ... 前期准备代码保持不变 ...
    scaler = amp.GradScaler()
    loss_meter = AverageMeter()
    
    # 获取损失函数权重 (建议在 cfg 中定义)
    alpha = getattr(cfg.MODEL, 'ORTH_WEIGHT', 0.1) 

    model.train()
    for epoch in range(1, epochs + 1):
        loss_meter.reset()
        scheduler.step(epoch)
        
        # 假设 data 包含 modal_id (0: Optical, 1: SAR)
        for n_iter, (img, pid, camid, target_view, modal_id) in enumerate(train_loader_pair):
            optimizer.zero_grad()
            img = img.to(local_rank)
            pid = pid.to(local_rank)
            modal_id = modal_id.to(local_rank)

            with amp.autocast(enabled=True):
                # 1. FAAM: 频域自适应增强 (仅在训练且混合模态时执行)
                # 逻辑：将 SAR 的幅度混合到 Optical 中
                if img.size(0) > 1:
                    opt_mask = (modal_id == 0)
                    sar_mask = (modal_id == 1)
                    if opt_mask.any() and sar_mask.any():
                        # 调用模型内部的 faam 逻辑或外部工具函数
                        img[opt_mask] = model.module.faam(img[opt_mask], img[sar_mask])

                # 2. Forward: 模型返回多个特征用于解耦和对齐
                # score: 分类概率, feat: 融合后的对齐特征, f_sh: 共享特征, f_sp: 特定特征
                score, feat, f_sh, f_sp = model(img, label=pid, modal_label=modal_id)

                # 3. Loss Calculation
                # 计算常规 ID 和 Triplet Loss
                loss_base = loss_func(score, feat, pid) 
                
                # 计算正交损失 (MDFE 核心约束)
                # 强制共享分支和特定分支关注不同的信息
                f_sh_norm = torch.nn.functional.normalize(f_sh, p=2, dim=1)
                f_sp_norm = torch.nn.functional.normalize(f_sp, p=2, dim=1)
                loss_orth = torch.mean(torch.abs(torch.mm(f_sh_norm, f_sp_norm.t())))
                
                # 总损失叠加
                total_loss = loss_base + alpha * loss_orth

            # 4. Backward
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(total_loss.item(), img.shape[0])


def do_train(cfg, model, center_criterion, train_loader, val_loader, optimizer, optimizer_center, scheduler, loss_func, num_query, local_rank):
    # ... (初始化代码保持不变) ...
    log_period = cfg.SOLVER.LOG_PERIOD
    scaler = amp.GradScaler()
    
    # 设定 FD-GNN 的正交损失权重
    alpha = getattr(cfg.MODEL, 'ORTH_WEIGHT', 0.1) 

    model.train()
    for epoch in range(1, epochs + 1):
        loss_meter.reset()
        scheduler.step(epoch)
        
        # 确保 Dataset 返回了 modal_id (0: Optical, 1: SAR)
        for n_iter, (img, pid, target_view, modal_id) in enumerate(train_loader):
            optimizer.zero_grad()
            if optimizer_center is not None:
                optimizer_center.zero_grad()
            
            img = img.to(local_rank)
            pid = pid.to(local_rank)
            modal_id = modal_id.to(local_rank)

            with amp.autocast(enabled=True):
                # --- [1. 模型前向传播] ---
                # FD-GNN 返回: 分类分, 融合特征, 共享特征, 特定特征
                score, feat, f_sh, f_sp = model(img, label=pid, modal_label=modal_id)

                # --- [2. 基础损失计算] ---
                # 包含原本的 CrossEntropy 和 Triplet Loss
                loss = loss_func(score, feat, pid)

                # --- [3. MDFE 正交解耦损失] ---
                # 强制 f_sh (身份相关) 与 f_sp (模态相关) 在特征空间正交
                f_sh_norm = torch.nn.functional.normalize(f_sh, p=2, dim=1)
                f_sp_norm = torch.nn.functional.normalize(f_sp, p=2, dim=1)
                # 计算互相关矩阵的 Frobenius 范数简化版
                loss_orth = torch.mean(torch.abs(torch.mm(f_sh_norm, f_sp_norm.t())))

                # --- [4. 总损失叠加] ---
                total_loss = loss + alpha * loss_orth

            # --- [5. 反向传播与优化] ---
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            
            if optimizer_center is not None:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
            
            scaler.update()


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath, img_wh) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            img_wh = img_wh.to(device)
            feat = model(img, cam_label=camids, img_wh=img_wh)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]
