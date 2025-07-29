import traceback
# 健壮的 advantage normalization 实现
def safe_advantage_normalization(batch, config, metrics):
    """
    安全的advantage normalization，确保在任何情况下都不会出错
    
    Args:
        batch: 批次数据
        config: 配置对象
        metrics: 指标字典
    """
    try:
        norm_root = getattr(config, "semantic_advantage", None)
        adv_norm_cfg = getattr(norm_root, "adv_norm", None) if norm_root else None

        # 早期退出条件
        if not adv_norm_cfg or not getattr(adv_norm_cfg, "enable", True):
            return  # 不做任何处理
        
        if "advantages" not in batch.batch:
            print("[WARNING] No advantages found in batch, skipping normalization")
            return

        level = getattr(adv_norm_cfg, "level", "batch")
        group_size = getattr(adv_norm_cfg, "group_size", None)
        use_mask_type = getattr(norm_root, "mask_type", "loss_mask")

        with torch.no_grad():
            adv = batch.batch["advantages"]
            
            # 基本健全性检查
            if adv.numel() == 0:
                print("[WARNING] Empty advantages tensor, skipping normalization")
                return
                
            if adv.dim() != 2:
                print(f"[WARNING] Unexpected advantages shape {adv.shape}, expected 2D tensor")
                return
                
            bs, resp_len = adv.shape
            
            if bs == 0 or resp_len == 0:
                print(f"[WARNING] Invalid batch dimensions bs={bs}, resp_len={resp_len}")
                return

            # 安全获取mask
            mask_all = None
            
            try:
                if use_mask_type == "loss_mask" and "loss_mask" in batch.batch:
                    loss_mask = batch.batch["loss_mask"]
                    
                    # 确保loss_mask是2D的
                    if loss_mask.dim() != 2:
                        print(f"[WARNING] loss_mask has unexpected shape {loss_mask.shape}")
                        mask_all = batch.batch.get("response_mask", torch.ones_like(adv)).bool()
                    else:
                        # 安全的切片操作
                        if loss_mask.shape[0] != bs:
                            print(f"[WARNING] loss_mask batch size mismatch: {loss_mask.shape[0]} vs {bs}")
                            mask_all = batch.batch.get("response_mask", torch.ones_like(adv)).bool()
                        elif loss_mask.shape[1] >= resp_len:
                            mask_all = loss_mask[:, -resp_len:].bool()
                        else:
                            # loss_mask太短，用零填充
                            print(f"[WARNING] loss_mask too short ({loss_mask.shape[1]} < {resp_len}), padding with zeros")
                            padding_size = resp_len - loss_mask.shape[1]
                            padding = torch.zeros(bs, padding_size, dtype=loss_mask.dtype, device=loss_mask.device)
                            padded_mask = torch.cat([padding, loss_mask], dim=1)
                            mask_all = padded_mask.bool()
                else:
                    # fallback到response_mask
                    if "response_mask" in batch.batch:
                        response_mask = batch.batch["response_mask"]
                        if response_mask.shape == adv.shape:
                            mask_all = response_mask.bool()
                        else:
                            print(f"[WARNING] response_mask shape mismatch: {response_mask.shape} vs {adv.shape}")
                            mask_all = torch.ones_like(adv).bool()
                    else:
                        print("[WARNING] No valid mask found, using all ones")
                        mask_all = torch.ones_like(adv).bool()
                        
            except Exception as e:
                print(f"[ERROR] Failed to get mask: {e}, using all ones")
                mask_all = torch.ones_like(adv).bool()

            # 确保mask形状正确
            if mask_all.shape != adv.shape:
                print(f"[WARNING] Mask shape mismatch: {mask_all.shape} vs {adv.shape}, using all ones")
                mask_all = torch.ones_like(adv).bool()

            # 检查是否有有效tokens
            valid_token_count = mask_all.sum().item()
            if valid_token_count == 0:
                print("[WARNING] No valid tokens found, skipping normalization")
                _record_default_metrics(metrics, level, 0)
                return

            # 安全的normalization
            norm_adv = adv.clone()
            
            if level == "batch":
                success = _safe_batch_normalization(adv, mask_all, norm_adv, metrics, level)
            elif level == "group":
                if group_size is None:
                    group_size = getattr(config.actor_rollout_ref.rollout, "n", 1)
                success = _safe_group_normalization(adv, mask_all, norm_adv, metrics, level, group_size, bs)
            else:
                print(f"[ERROR] Unknown normalization level: {level}")
                return

            if success:
                # 最终安全检查
                if torch.isnan(norm_adv).any() or torch.isinf(norm_adv).any():
                    print("[ERROR] NaN/Inf detected in normalized advantages, reverting to original")
                    _record_default_metrics(metrics, level, valid_token_count)
                else:
                    # 安全更新
                    batch.batch["advantages"] = norm_adv
                    print(f"[INFO] Advantage normalization successful: {valid_token_count} tokens processed")
            else:
                _record_default_metrics(metrics, level, valid_token_count)

    except Exception as e:
        print(f"[ERROR] Exception in advantage normalization: {e}")
        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        # 确保不影响训练继续
        _record_default_metrics(metrics, getattr(adv_norm_cfg, "level", "batch"), 0)


def _safe_batch_normalization(adv, mask_all, norm_adv, metrics, level):
    """安全的batch级别normalization"""
    try:
        valid_adv = adv[mask_all]
        
        if valid_adv.numel() == 0:
            print("[WARNING] No valid advantages for batch normalization")
            return False
            
        # 使用更稳定的统计量计算
        if valid_adv.numel() == 1:
            med = valid_adv[0]
        else:
            med = torch.median(valid_adv)
            
        # 检查median是否有效
        if torch.isnan(med) or torch.isinf(med):
            print(f"[WARNING] Invalid median value: {med}")
            return False
            
        # 应用normalization
        norm_adv[mask_all] = adv[mask_all] - med
        
        # 记录指标
        tokens_normed = int(mask_all.sum().item())
        _record_batch_metrics(metrics, norm_adv, mask_all, med.item(), tokens_normed)
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Batch normalization failed: {e}")
        return False


def _safe_group_normalization(adv, mask_all, norm_adv, metrics, level, group_size, bs):
    """安全的group级别normalization"""
    try:
        device = adv.device
        group_ids = torch.arange(bs, device=device) // group_size
        unique_groups = group_ids.unique()
        
        tokens_normed = 0
        med_list = []
        successful_groups = 0
        
        for gid in unique_groups:
            try:
                g_mask_sample = (group_ids == gid).unsqueeze(1)
                g_mask = g_mask_sample & mask_all
                
                if not g_mask.any():
                    continue
                    
                g_adv = adv[g_mask]
                if g_adv.numel() == 0:
                    continue
                    
                # 计算group median
                if g_adv.numel() == 1:
                    g_med = g_adv[0]
                else:
                    g_med = torch.median(g_adv)
                    
                if torch.isnan(g_med) or torch.isinf(g_med):
                    print(f"[WARNING] Invalid median for group {gid}: {g_med}")
                    continue
                    
                # 应用normalization
                norm_adv[g_mask] = adv[g_mask] - g_med
                
                med_list.append(g_med)
                tokens_normed += int(g_adv.numel())
                successful_groups += 1
                
            except Exception as e:
                print(f"[WARNING] Failed to process group {gid}: {e}")
                continue
        
        if successful_groups == 0:
            print("[WARNING] No groups successfully processed")
            return False
            
        # 记录指标
        med_mean = torch.stack(med_list).mean().item() if med_list else 0.0
        _record_group_metrics(metrics, norm_adv, mask_all, med_mean, tokens_normed, 
                            len(unique_groups), successful_groups)
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Group normalization failed: {e}")
        return False


def _record_default_metrics(metrics, level, tokens_count):
    """记录默认指标"""
    metrics.update({
        "adv_norm/level": level,
        "adv_norm/groups": 0,
        "adv_norm/tokens_normed": tokens_count,
        "adv_norm/zero_groups": 0,
        "adv_norm/median_mean": 0.0,
        "adv_norm/pos_tokens": 0,
        "adv_norm/neg_tokens": 0,
        "adv_norm/zero_tokens": tokens_count,
        "adv_norm/pos_sequences": 0,
        "adv_norm/neg_sequences": 0,
        "adv_norm/zero_sequences": 0,
        "adv_norm/neg_token_ratio": 0.0,
    })


def _record_batch_metrics(metrics, norm_adv, mask_all, med_value, tokens_normed):
    """记录batch级别指标"""
    pos_tok = int((norm_adv[mask_all] > 0).sum().item())
    neg_tok = int((norm_adv[mask_all] < 0).sum().item())
    zero_tok = int((norm_adv[mask_all] == 0).sum().item())
    
    seq_sum = (norm_adv * mask_all).sum(dim=1)
    pos_seq = int((seq_sum > 0).sum().item())
    neg_seq = int((seq_sum < 0).sum().item())
    zero_seq = int((seq_sum == 0).sum().item())
    
    metrics.update({
        "adv_norm/level": "batch",
        "adv_norm/groups": 1,
        "adv_norm/tokens_normed": tokens_normed,
        "adv_norm/zero_groups": 0,
        "adv_norm/median_mean": med_value,
        "adv_norm/pos_tokens": pos_tok,
        "adv_norm/neg_tokens": neg_tok,
        "adv_norm/zero_tokens": zero_tok,
        "adv_norm/pos_sequences": pos_seq,
        "adv_norm/neg_sequences": neg_seq,
        "adv_norm/zero_sequences": zero_seq,
        "adv_norm/neg_token_ratio": neg_tok / max(1, pos_tok + neg_tok),
    })


def _record_group_metrics(metrics, norm_adv, mask_all, med_mean, tokens_normed, 
                         total_groups, successful_groups):
    """记录group级别指标"""
    pos_tok = int((norm_adv[mask_all] > 0).sum().item())
    neg_tok = int((norm_adv[mask_all] < 0).sum().item())
    zero_tok = int((norm_adv[mask_all] == 0).sum().item())
    
    seq_sum = (norm_adv * mask_all).sum(dim=1)
    pos_seq = int((seq_sum > 0).sum().item())
    neg_seq = int((seq_sum < 0).sum().item())
    zero_seq = int((seq_sum == 0).sum().item())
    
    metrics.update({
        "adv_norm/level": "group",
        "adv_norm/groups": total_groups,
        "adv_norm/tokens_normed": tokens_normed,
        "adv_norm/zero_groups": total_groups - successful_groups,
        "adv_norm/median_mean": med_mean,
        "adv_norm/pos_tokens": pos_tok,
        "adv_norm/neg_tokens": neg_tok,
        "adv_norm/zero_tokens": zero_tok,
        "adv_norm/pos_sequences": pos_seq,
        "adv_norm/neg_sequences": neg_seq,
        "adv_norm/zero_sequences": zero_seq,
        "adv_norm/neg_token_ratio": neg_tok / max(1, pos_tok + neg_tok),
    })


