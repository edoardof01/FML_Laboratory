# losses.py
"""Custom loss functions for contrastive learning."""
import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning with optional Jaccard similarity for multi-label.
    """

    def __init__(
        self,
        temperature=0.07,
        contrast_mode="all",
        base_temperature=0.07,
        use_jaccard=False,
        jaccard_threshold=0.5,
    ):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.use_jaccard = use_jaccard
        self.jaccard_threshold = jaccard_threshold

    def forward(self, features, labels=None, mask=None):
        """
        Args:
            features: hidden vector of shape [bsz, n_views, dim].
            labels: ground truth of shape [bsz] or [bsz, num_classes].
            mask: contrastive mask of shape [bsz, bsz].
        Returns:
            A loss scalar.
        """
        device = features.device

        if features.dim() < 3:
            raise ValueError("`features` needs at least 3 dimensions [bsz, n_views, dim]")
        if features.dim() > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            if labels.dim() == 2:
                # Multi-label: binary vectors [bsz, num_classes]
                labels = labels.float()
                if self.use_jaccard:
                    intersection = torch.mm(labels, labels.T)
                    sum_labels = labels.sum(1, keepdim=True)
                    union = sum_labels + sum_labels.T - intersection
                    union = torch.clamp(union, min=1e-6)
                    jaccard = intersection / union
                    mask = (jaccard >= self.jaccard_threshold).float().to(device)
                else:
                    mask = (torch.mm(labels, labels.T) > 0).float().to(device)
            else:
                # Single-label
                labels = labels.contiguous().view(-1, 1)
                if labels.shape[0] != batch_size:
                    raise ValueError("Num of labels does not match num of features")
                mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError(f"Unknown mode: {self.contrast_mode}")

        # Compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )

        # Numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Tile mask
        mask = mask.repeat(anchor_count, contrast_count)

        # Mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # Mean of log-likelihood over positive pairs
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1.0, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # Loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
