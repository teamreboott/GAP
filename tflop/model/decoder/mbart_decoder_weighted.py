"""Geometry-Aware Pointer (GAP) Loss.

Reweights the pointer cross-entropy by Manhattan distance d between the
ground-truth cell and each negative candidate in the table grid:

    w(d) = max(alpha / 2**d, 0.5),    d >= 1,  alpha = 8

giving weights 4, 2, 1, 0.5 for d = 1, 2, 3, 4 and 0.5 for d >= 5.
Negative weights are then mass-normalised to sum to kappa, so the total
negative mass is held constant and only its spatial distribution changes.

Note: Eq. (7) of the paper prints this as alpha / 2**(d-1); the exponent
there is a typo. The weights quoted in the same paragraph (4, 2, 1 for
d = 1, 2, 3) and Fig. 4 both match alpha / 2**d, which is what this
implementation and the released checkpoints use.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def integrate_weighted_ce_into_decoder(
    is_not_empty_pred,
    is_not_empty_label,
    valid_coords_tmp,
    use_adjacent_penalty=True,
    adjacent_penalty_config=None
):
    """
    Integration function for mbart_decoder.py
    
    This replaces the standard CrossEntropyLoss in get_tag2coord_ptr_loss
    """
    
    if not use_adjacent_penalty:
        # Standard CrossEntropy (original behavior)
        return nn.CrossEntropyLoss()(
            torch.transpose(is_not_empty_pred, 0, 1)[valid_coords_tmp],
            torch.argmax(
                torch.transpose(is_not_empty_label, 0, 1)[valid_coords_tmp],
                dim=-1,
            ),
        )
    
    # Apply adjacent tag penalty
    if adjacent_penalty_config is None:
        adjacent_penalty_config = {
            'max_distance': 2,
            'weights': {1: 2.0, 2: 1.5}
        }
    
    # Prepare inputs
    # is_not_empty_pred: [num_text, num_bbox]
    # is_not_empty_label: [num_text, num_bbox]
    
    # Filter by valid coordinates
    pred_filtered = torch.transpose(is_not_empty_pred, 0, 1)[valid_coords_tmp]  # [num_valid_bbox, num_text]
    label_filtered = torch.transpose(is_not_empty_label, 0, 1)[valid_coords_tmp]  # [num_valid_bbox, num_text]
    
    # Get target indices for each valid bbox
    target_indices = torch.argmax(label_filtered, dim=1)  # [num_valid_bbox]
    
    # Create weight matrix based on adjacent penalties
    num_valid_bbox = pred_filtered.shape[0]
    num_text = pred_filtered.shape[1]
    device = pred_filtered.device
    
    total_loss = 0
    
    for bbox_idx in range(num_valid_bbox):
        target_tag = target_indices[bbox_idx].item()
        
        # Create weights for this bbox
        weights = torch.ones(num_text, device=device)
        
        # Apply adjacent penalties
        for distance, weight in adjacent_penalty_config['weights'].items():
            for offset in [-distance, distance]:
                adjacent_tag = target_tag + offset
                if 0 <= adjacent_tag < num_text:
                    weights[adjacent_tag] = weight
        
        # Compute weighted loss for this bbox
        bbox_loss = F.cross_entropy(
            pred_filtered[bbox_idx:bbox_idx+1],  # [1, num_text]
            target_indices[bbox_idx:bbox_idx+1],  # [1]
            weight=weights,
            reduction='mean'
        )
        
        total_loss += bbox_loss
    
    return total_loss / num_valid_bbox if num_valid_bbox > 0 else 0


def integrate_weighted_ce_into_decoder_v2(
    is_not_empty_pred,     # [num_text, num_bbox], upstream에서 temperature 미적용 권장
    is_not_empty_label,    # [num_text, num_bbox], one-hot
    valid_coords_tmp,      # [num_bbox]
    use_adjacent_penalty=True,
    adjacent_penalty_config=None
):
    """
    Spatial-aware Weighted Cross-Entropy with Mass Normalization.
    Core features: spatial weights and mass normalization.
    Optional features: adaptive margin, focal loss, difficulty weighting.
    
    Args:
        is_not_empty_pred: [num_text, num_bbox] predictions (before temperature)
        is_not_empty_label: [num_text, num_bbox] one-hot labels
        valid_coords_tmp: [num_bbox] boolean mask for valid coordinates
        use_adjacent_penalty: whether to use spatial weighting
        adjacent_penalty_config: dict with configuration:
            Required for spatial weights:
                - row_id: row indices of each candidate
                - col_id: column indices of each candidate
            Core parameters:
                - use_spatial_weights: bool (default: True)
                - use_mass_normalization: bool (default: True)
                - temperature: float (default: 0.1)
                - use_logitnorm: bool (default: True)
                - kappa: float (default: 1.0) - mass normalization constant
            Spatial weight parameters:
                - decay_rate: float (default: 0.5) - spatial decay
                - min_weight: float (default: 0.1) - minimum weight
                - immediate_boost: float (default: 2.0) - boost for distance=1
            Optional features (disabled by default):
                - use_dynamic_kappa: bool (default: False)
                - use_ccp_bias: bool (default: False)
                - use_adaptive_margin: bool (default: False)
                - use_focal_loss: bool (default: False)
                - use_difficulty_weight: bool (default: False)
    
    Returns:
        loss: scalar loss value
    """
    import math
    
    pred_f = torch.transpose(is_not_empty_pred, 0, 1)[valid_coords_tmp]   # [Bv, C]
    lab_f  = torch.transpose(is_not_empty_label, 0, 1)[valid_coords_tmp]  # [Bv, C]
    Bv, C  = pred_f.shape
    if Bv == 0:
        return pred_f.new_tensor(0.0)

    cfg = adjacent_penalty_config or {}
    device = pred_f.device
    eps = 1e-8

    # (1) Temperature scaling
    temp = float(cfg.get('temperature', 0.1))
    pred_f = pred_f / max(temp, 1e-6)

    # (2) LogitNorm (optional but recommended)
    if cfg.get('use_logitnorm', True):
        mu  = pred_f.mean(dim=1, keepdim=True)
        std = pred_f.std(dim=1, keepdim=True).clamp_min(1e-6)
        pred_f = (pred_f - mu) / std

    # Get target indices
    tgt = lab_f.argmax(dim=1)  # [Bv]

    # (3) Spatial weights - CORE FEATURE
    dist = None
    use_spatial = cfg.get('use_spatial_weights', True)
    if use_spatial and use_adjacent_penalty and ('row_id' in cfg) and ('col_id' in cfg):
        row_id = torch.as_tensor(cfg['row_id'], device=device, dtype=torch.long)  # [C]
        col_id = torch.as_tensor(cfg['col_id'], device=device, dtype=torch.long)  # [C]
        tr = row_id[tgt].unsqueeze(1)  # [Bv,1]
        tc = col_id[tgt].unsqueeze(1)
        r  = row_id.unsqueeze(0)       # [1,C]
        c  = col_id.unsqueeze(0)
        dist = (tr - r).abs() + (tc - c).abs()  # [Bv,C] Manhattan distance

        # GAP spatial weights: w(d) = max(alpha / 2**d, min_weight)
        # alpha=8 -> 4, 2, 1, 0.5 for d=1..4; floored at 0.5 for d>=5.
        alpha = float(cfg.get('alpha', 8.0))
        min_w = float(cfg.get('min_weight', 0.5))
        w = torch.clamp(alpha * torch.pow(0.5, dist.float()), min=min_w)
    else:
        w = pred_f.new_ones((Bv, C))

    # (4) Mass normalization - CORE FEATURE
    if cfg.get('use_mass_normalization', True):
        base_kappa = float(cfg.get('kappa', 1.0))
        
        # Optional: Dynamic kappa based on confidence
        if cfg.get('use_dynamic_kappa', False):
            with torch.no_grad():
                p = torch.softmax(pred_f, dim=1)
                pt = p[torch.arange(Bv, device=device), tgt]
                dyn_mul = (1.0 + (1.0 - pt))  # 1.0 ~ 2.0 range
                dynamic_kappa = (base_kappa * dyn_mul).clamp(base_kappa*0.5, base_kappa*2.0).unsqueeze(1)
        else:
            dynamic_kappa = torch.full((Bv, 1), base_kappa, device=device)
        
        # Mass-normalize negative weights to sum to kappa
        mask_pos = torch.zeros_like(w, dtype=torch.bool)
        mask_pos[torch.arange(Bv, device=device), tgt] = True
        w_neg = w.masked_fill(mask_pos, 0.0)
        sum_w = w_neg.sum(dim=1, keepdim=True).clamp_min(eps)
        w = w_neg * (dynamic_kappa / sum_w)
    else:
        # No mass normalization - just mask out positive class
        mask_pos = torch.zeros_like(w, dtype=torch.bool)
        mask_pos[torch.arange(Bv, device=device), tgt] = True
        w = w.masked_fill(mask_pos, 0.0)

    # (5) Optional: Adaptive margin
    if cfg.get('use_adaptive_margin', False):
        base_margin = float(cfg.get('cos_margin', 0.1))
        if dist is not None:
            avg_dist = dist.float().mean(dim=1)  # [Bv]
            adaptive_margin = base_margin * (1.0 + 1.0 / (avg_dist + 1.0))
            pred_f[torch.arange(Bv, device=device), tgt] -= adaptive_margin
        else:
            pred_f[torch.arange(Bv, device=device), tgt] -= base_margin

    # (6) Optional: CCP bias
    if cfg.get('use_ccp_bias', False):
        pi = 1.0 / float(C)
        if cfg.get('use_mass_normalization', True):
            # b_neg = log((1 - pi) / (kappa * pi))
            kappa_val = dynamic_kappa.squeeze(1) if cfg.get('use_dynamic_kappa', False) else base_kappa
            b_neg = torch.log1p(torch.tensor(-pi, device=device)) - torch.log(kappa_val * pi)
            if not cfg.get('use_dynamic_kappa', False):
                b_neg = b_neg.expand(Bv)
            b_neg = b_neg.unsqueeze(1)  # [Bv,1]
        else:
            b_neg = torch.zeros((Bv, 1), device=device)
    else:
        b_neg = torch.zeros((Bv, 1), device=device)

    # (7) Compute loss with logsumexp for stability
    pos = pred_f[torch.arange(Bv, device=device), tgt]    # [Bv]
    neg = pred_f + b_neg
    
    # Mask out positive class for negative term
    mask_pos = torch.zeros_like(neg, dtype=torch.bool)
    mask_pos[torch.arange(Bv, device=device), tgt] = True
    neg = neg.masked_fill(mask_pos, float('-inf'))

    log_w  = torch.log(w.clamp_min(eps))
    lse_neg= torch.logsumexp(log_w + neg, dim=1)          # [Bv]
    log_den= torch.logaddexp(pos, lse_neg)                # [Bv]
    loss = -(pos - log_den)

    # (8) Optional: Focal loss
    if cfg.get('use_focal_loss', False):
        gamma = float(cfg.get('focal_gamma', 2.0))
        with torch.no_grad():
            p = torch.softmax(pred_f, dim=1)
            pt = p[torch.arange(Bv, device=device), tgt]
            focal_w = (1.0 - pt).pow(gamma)
        loss = loss * focal_w

    # (9) Optional: Difficulty weighting
    if cfg.get('use_difficulty_weight', False):
        with torch.no_grad():
            p = torch.softmax(pred_f, dim=1)
            log_p = torch.log(p + 1e-10)
            entropy = -(p * log_p).sum(dim=1)
            max_entropy = math.log(C) if C > 1 else 1.0
            difficulty = (entropy / max_entropy).clamp(0.1, 1.0)
        loss = loss * difficulty

    return loss.mean()
