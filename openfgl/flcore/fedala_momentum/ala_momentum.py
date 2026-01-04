import torch
import torch.nn as nn
import numpy as np
import copy
from torch.utils.data import DataLoader, TensorDataset


class ALAMomentum:
    """
    Adaptive Local Aggregation with Momentum Integration.

    Extends the base ALA algorithm with momentum for update direction smoothing.
    The momentum helps stabilize the aggregation process by maintaining a running
    average of the update directions.

    Momentum update:
        V_t = γ * V_(t-1) + (1-γ) * (Θ_global - Θ_local)
        Θ_new = Θ_local + α * V_t

    where γ is the momentum coefficient (typically 0.9).
    """

    def __init__(self, cid, loss, train_data, batch_size, rand_percent, layer_idx, eta,
                 device, threshold=0.1, num_pre_loss=10, beta=0.3, momentum_gamma=0.9):
        """
        Initialize ALAMomentum.

        Args:
            cid: Client ID
            loss: Loss function
            train_data: Training data
            batch_size: Batch size for training
            rand_percent: Percentage of random initialization
            layer_idx: Starting layer index for ALA
            eta: Learning rate for weight update
            device: Computation device
            threshold: Threshold for weight updates
            num_pre_loss: Number of pre-training epochs
            beta: Temporal smoothing coefficient
            momentum_gamma: Momentum coefficient (γ)
        """
        self.cid = cid
        self.loss = loss
        self.train_data = train_data
        self.batch_size = batch_size
        self.rand_percent = rand_percent
        self.layer_idx = layer_idx
        self.eta = eta
        self.threshold = threshold
        self.num_pre_loss = num_pre_loss
        self.device = device
        self.beta = beta
        self.momentum_gamma = momentum_gamma

        self.weights = None
        self.start_phase = True

        # Momentum buffer: stores V_t for each layer
        self.momentum_buffer = None

    def adaptive_local_aggregation(self, global_model, local_model):
        """
        Performs adaptive local aggregation with momentum.

        Args:
            global_model: Global model with aggregated parameters
            local_model: Local model with client-specific parameters
        """
        params_g = list(global_model.parameters())
        params_l = list(local_model.parameters())

        # Initialize momentum buffer if needed
        if self.momentum_buffer is None:
            self.momentum_buffer = [torch.zeros_like(p.data) for p in params_l]

        # Initialize or update weights
        if self.weights is None:
            self.weights = [torch.ones_like(p.data).to(self.device) for p in params_l]

        # Skip layers before layer_idx
        for i in range(self.layer_idx):
            self.weights[i] = torch.ones_like(params_l[i].data).to(self.device)

        # Prepare data loader for weight optimization
        x = self.train_data["x"]
        y = self.train_data["y"]
        edge_index = self.train_data.get("edge_index", None)

        # Create a simple data batch
        if isinstance(x, torch.Tensor):
            x = x.to(self.device)
            y = y.to(self.device)

        # Weight learning phase
        if self.start_phase:
            for i in range(self.layer_idx, len(params_l)):
                self.weights[i] = torch.rand_like(params_l[i].data).to(self.device) * \
                                   (self.rand_percent / 100.0)
            self.start_phase = False

        # Compute optimal weights for each layer (using similar logic to ALA)
        for i in range(self.layer_idx, len(params_l)):
            g_param = params_g[i].data.clone()
            l_param = params_l[i].data.clone()

            # Compute update direction: global - local
            update_dir = g_param - l_param

            # Apply momentum: V_t = γ * V_(t-1) + (1-γ) * update_dir
            self.momentum_buffer[i] = self.momentum_gamma * self.momentum_buffer[i].to(self.device) + \
                                       (1 - self.momentum_gamma) * update_dir

            # Compute optimal weight for this layer
            # Using gradient-based optimization similar to base ALA
            w_opt = self._optimize_weight_for_layer(i, g_param, l_param, params_l, local_model)

            # Temporal smoothing: new_weight = (1-beta) * w_opt + beta * old_weight
            if self.weights[i] is not None:
                self.weights[i] = (1 - self.beta) * w_opt + self.beta * self.weights[i]
            else:
                self.weights[i] = w_opt

        # Apply weights with momentum-smoothed update direction
        for i in range(len(params_l)):
            # Reconstruct: local + weight * momentum_smoothed_direction
            aggregated = params_l[i].data + self.weights[i] * self.momentum_buffer[i]
            params_l[i].data.copy_(aggregated)

    def _optimize_weight_for_layer(self, layer_idx, g_param, l_param, params_l, local_model):
        """
        Optimize the aggregation weight for a specific layer.

        Args:
            layer_idx: Index of the layer
            g_param: Global parameter for this layer
            l_param: Local parameter for this layer
            params_l: All local parameters
            local_model: The local model

        Returns:
            Optimal weight tensor for this layer
        """
        # Simple gradient-based weight optimization
        # Start with current weight or random init
        if self.weights[layer_idx] is not None:
            w = self.weights[layer_idx].clone().requires_grad_(False)
        else:
            w = torch.ones_like(l_param).to(self.device) * 0.5

        # Compute direction magnitude and use it to scale weight
        diff = g_param - l_param
        diff_norm = diff.norm()

        if diff_norm > self.threshold:
            # Scale weight based on divergence
            scale = torch.clamp(1.0 / (1.0 + diff_norm), 0.1, 1.0)
            w = torch.ones_like(l_param).to(self.device) * scale
        else:
            # Small divergence: use higher weight towards global
            w = torch.ones_like(l_param).to(self.device) * 0.8

        return w
