# Copyright (c) OpenMMLab. All rights reserved.
# Adapted from NVIDIA's semantic-segmentation repo (MIT License):
# https://github.com/NVIDIA/semantic-segmentation
#
# Region Mutual Information Loss for Semantic Segmentation
# Reference: Zhao et al., "Region Mutual Information Loss for Semantic
# Segmentation", NeurIPS 2019. https://arxiv.org/abs/1910.12037

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS


def _map_get_pairs(labels_4D, probs_4D, radius=3):
    """Extract local region feature pairs for RMI computation.

    For each pixel, collect features from a local region of size
    ``radius x radius``.  Returns tensors of shape
    ``(N, C, radius*radius, h', w')`` where ``h', w'`` are the spatial
    dimensions reduced by ``radius - 1``.

    Args:
        labels_4D (torch.Tensor): One-hot labels ``(N, C, H, W)``.
        probs_4D (torch.Tensor): Predicted probabilities ``(N, C, H, W)``.
        radius (int): Side length of the local region. Default: 3.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - la_vectors: label vectors ``(N, C, radius*radius, h', w')``
            - pr_vectors: prediction vectors ``(N, C, radius*radius, h', w')``
    """
    h, w = labels_4D.shape[2], labels_4D.shape[3]
    new_h, new_w = h - radius + 1, w - radius + 1
    la_ns = []
    pr_ns = []
    for y in range(radius):
        for x in range(radius):
            la_ns.append(labels_4D[:, :, y:y + new_h, x:x + new_w])
            pr_ns.append(probs_4D[:, :, y:y + new_h, x:x + new_w])
    # Stack along a new "region" dimension → (N, C, R*R, h', w')
    la_vectors = torch.stack(la_ns, dim=2)
    pr_vectors = torch.stack(pr_ns, dim=2)
    return la_vectors, pr_vectors


def _log_det_by_cholesky(matrix):
    """Compute log-determinant via Cholesky decomposition.

    Args:
        matrix (torch.Tensor): Positive-definite matrices of shape
            ``(..., D, D)``.

    Returns:
        torch.Tensor: Log-determinant values of shape ``(...)``.
    """
    chol = torch.linalg.cholesky(matrix)
    diag = torch.diagonal(chol, dim1=-2, dim2=-1)
    return 2.0 * torch.sum(torch.log(diag + 1e-8), dim=-1)


def rmi_lower_bound(labels_4D, probs_4D, num_classes, rmi_radius=3,
                    rmi_pool_way=1, rmi_pool_size=3, rmi_pool_stride=3):
    """Compute the Region Mutual Information lower bound.

    Steps:
        1. Pool predictions and labels for efficiency.
        2. Extract local region pairs of size ``rmi_radius``.
        3. Compute covariance matrices.
        4. Compute conditional entropy ``H(P|T)`` via Schur complement.
        5. Return the mean conditional entropy (to be minimised).

    Args:
        labels_4D (torch.Tensor): One-hot label tensor ``(N, C, H, W)``.
        probs_4D (torch.Tensor): Softmax probability tensor ``(N, C, H, W)``.
        num_classes (int): Number of semantic classes.
        rmi_radius (int): Radius (side length) for local regions. Default: 3.
        rmi_pool_way (int): Pooling method (0=max, 1=avg, 2=stride).
            Default: 1.
        rmi_pool_size (int): Pooling kernel size. Default: 3.
        rmi_pool_stride (int): Pooling stride. Default: 3.

    Returns:
        torch.Tensor: Scalar RMI loss (conditional entropy lower bound).
    """
    # --- 1. down-sample for efficiency ---
    if rmi_pool_stride > 1:
        if rmi_pool_way == 0:
            labels_4D = F.max_pool2d(
                labels_4D, kernel_size=rmi_pool_size,
                stride=rmi_pool_stride,
                padding=rmi_pool_size // 2)
            probs_4D = F.max_pool2d(
                probs_4D, kernel_size=rmi_pool_size,
                stride=rmi_pool_stride,
                padding=rmi_pool_size // 2)
        elif rmi_pool_way == 1:
            labels_4D = F.avg_pool2d(
                labels_4D, kernel_size=rmi_pool_size,
                stride=rmi_pool_stride,
                padding=rmi_pool_size // 2)
            probs_4D = F.avg_pool2d(
                probs_4D, kernel_size=rmi_pool_size,
                stride=rmi_pool_stride,
                padding=rmi_pool_size // 2)
        elif rmi_pool_way == 2:
            labels_4D = labels_4D[:, :, ::rmi_pool_stride, ::rmi_pool_stride]
            probs_4D = probs_4D[:, :, ::rmi_pool_stride, ::rmi_pool_stride]

    # --- 2. extract local region feature pairs ---
    la_vectors, pr_vectors = _map_get_pairs(
        labels_4D, probs_4D, radius=rmi_radius)
    # Shape: (N, C, D, h, w)  where D = rmi_radius^2

    N, C, D = la_vectors.shape[:3]
    # Flatten spatial dims → (N, C, D, M)  where M = h*w
    la_vectors = la_vectors.reshape(N, C, D, -1)
    pr_vectors = pr_vectors.reshape(N, C, D, -1)
    M = la_vectors.shape[-1]

    # --- 3. compute covariance matrices ---
    # Centre the vectors (subtract spatial mean)
    la_mean = la_vectors.mean(dim=-1, keepdim=True)
    pr_mean = pr_vectors.mean(dim=-1, keepdim=True)
    la_vectors = la_vectors - la_mean
    pr_vectors = pr_vectors - pr_mean

    # Sigma_label: (N, C, D, D)
    la_cov = torch.matmul(la_vectors, la_vectors.transpose(2, 3)) / M
    # Sigma_pred:  (N, C, D, D)
    pr_cov = torch.matmul(pr_vectors, pr_vectors.transpose(2, 3)) / M
    # Cross-covariance Sigma_{pred,label}: (N, C, D, D)
    pr_la_cov = torch.matmul(pr_vectors, la_vectors.transpose(2, 3)) / M

    # --- 4. conditional covariance via Schur complement ---
    # Sigma_{P|T} = Sigma_P - Sigma_{P,T} Sigma_T^{-1} Sigma_{T,P}
    eye = torch.eye(D, dtype=la_cov.dtype, device=la_cov.device)
    eye = eye.unsqueeze(0).unsqueeze(0)  # (1, 1, D, D)

    la_cov_reg = la_cov + eye * 1e-5
    la_cov_inv = torch.inverse(la_cov_reg)

    appro_var = pr_cov - torch.matmul(
        pr_la_cov, torch.matmul(la_cov_inv, pr_la_cov.transpose(2, 3)))
    # Ensure positive-definite
    appro_var = appro_var + eye * 1e-6

    # --- 5. loss = 0.5 * sum_c log det(Sigma_{P|T}) ---
    # This is the conditional entropy H(P|T) (up to constants), which
    # equals H(P) - MI(P,T).  Minimising it maximises MI.
    rmi_per_class = 0.5 * _log_det_by_cholesky(appro_var)  # (N, C)
    rmi_per_class = rmi_per_class.mean(dim=0)  # (C,)

    loss = rmi_per_class.sum() / num_classes
    return loss


@MODELS.register_module()
class RMILoss(nn.Module):
    """Region Mutual Information Loss.

    This loss maximises the mutual information between predicted
    probabilities and ground-truth labels within local regions,
    providing a structured loss signal complementary to per-pixel
    cross-entropy.

    Reference:
        Zhao et al., "Region Mutual Information Loss for Semantic
        Segmentation", NeurIPS 2019.

    Args:
        num_classes (int): Number of semantic classes. Default: 19.
        rmi_radius (int): Side length of the local region. Default: 3.
        rmi_pool_way (int): Down-sampling strategy before RMI computation
            (0 = max-pool, 1 = avg-pool, 2 = stride). Default: 1.
        rmi_pool_size (int): Pooling kernel size. Default: 3.
        rmi_pool_stride (int): Pooling stride. Default: 3.
        loss_weight (float): Weight of this loss term. Default: 1.0.
        loss_name (str): Name used by the runner to aggregate losses.
            Must start with ``loss_``. Default: ``'loss_rmi'``.
        ignore_index (int): Label value to ignore. Default: 255.
    """

    def __init__(self,
                 num_classes=19,
                 rmi_radius=3,
                 rmi_pool_way=1,
                 rmi_pool_size=3,
                 rmi_pool_stride=3,
                 loss_weight=1.0,
                 loss_name='loss_rmi',
                 ignore_index=255):
        super().__init__()
        self.num_classes = num_classes
        self.rmi_radius = rmi_radius
        self.rmi_pool_way = rmi_pool_way
        self.rmi_pool_size = rmi_pool_size
        self.rmi_pool_stride = rmi_pool_stride
        self.loss_weight = loss_weight
        self._loss_name = loss_name
        self.ignore_index = ignore_index

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=255,
                **kwargs):
        """Forward computation.

        Args:
            cls_score (torch.Tensor): Logits of shape ``(N, C, H, W)``.
            label (torch.Tensor): Ground-truth of shape ``(N, H, W)``
                with integer class labels.
            weight: Unused (kept for API compatibility).
            avg_factor: Unused.
            reduction_override: Unused.
            ignore_index (int): Label value to ignore. Default: 255.

        Returns:
            torch.Tensor: Scalar RMI loss.
        """
        ignore_index = self.ignore_index

        # ---- resize label to match cls_score if needed ----
        if cls_score.shape[2:] != label.shape[1:]:
            label = F.interpolate(
                label.unsqueeze(1).float(),
                size=cls_score.shape[2:],
                mode='nearest').squeeze(1).long()

        # ---- force fp32 math for inverse/cholesky under AMP ----
        input_dtype = cls_score.dtype
        with torch.cuda.amp.autocast(enabled=False):
            cls_score = cls_score.float()

            # ---- softmax probabilities ----
            probs = F.softmax(cls_score, dim=1)

            # ---- build valid mask & one-hot labels ----
            valid_mask = (label >= 0) & (label < self.num_classes)
            if ignore_index is not None:
                valid_mask = valid_mask & (label != ignore_index)
            label_clamped = label.clone()
            label_clamped[~valid_mask] = 0

            onehot = F.one_hot(
                label_clamped, self.num_classes).permute(0, 3, 1, 2).float()
            # Zero-out ignored positions
            onehot = onehot * valid_mask.unsqueeze(1).float()

            # Mask predictions too
            probs = probs * valid_mask.unsqueeze(1).float()

            # ---- RMI lower bound (always in float32) ----
            loss = rmi_lower_bound(
                onehot, probs,
                num_classes=self.num_classes,
                rmi_radius=self.rmi_radius,
                rmi_pool_way=self.rmi_pool_way,
                rmi_pool_size=self.rmi_pool_size,
                rmi_pool_stride=self.rmi_pool_stride)

        return (self.loss_weight * loss).to(input_dtype)

    @property
    def loss_name(self):
        """Loss Name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
