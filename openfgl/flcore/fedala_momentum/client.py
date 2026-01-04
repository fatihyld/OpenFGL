import torch
import torch.nn as nn
import copy
from openfgl.flcore.base import BaseClient
from openfgl.flcore.fedala.ala import ALA
from openfgl.flcore.fedala_momentum.fedala_momentum_config import config


class FedALAMomentumClient(BaseClient):
    """
    FedALAMomentumClient combines FedALA-S with momentum integration for update direction.

    This implementation:
    1. Uses ALA for adaptive layer-wise aggregation between local and global models
    2. Applies temporal smoothing (beta) to stabilize aggregation weights
    3. Maintains momentum buffer for smoothing update directions over rounds

    Momentum update:
        V_t = γ * V_(t-1) + (1-γ) * (Θ_global - Θ_local)
        After ALA: Θ_new = Θ_local + α * V_t
    """

    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(FedALAMomentumClient, self).__init__(args, client_id, data, data_dir, message_pool, device)

        # Set default FedALA args if not present
        if not hasattr(self.args, 'rand_percent'): self.args.rand_percent = 80
        if not hasattr(self.args, 'layer_idx'): self.args.layer_idx = 0
        if not hasattr(self.args, 'eta'): self.args.eta = 1.0
        if not hasattr(self.args, 'threshold'): self.args.threshold = 0.1
        if not hasattr(self.args, 'num_pre_loss'): self.args.num_pre_loss = 10

        # FedALA-S: temporal smoothing with beta
        if not hasattr(self.args, 'ala_beta') or self.args.ala_beta == 0.0:
            self.args.ala_beta = 0.3

        # Momentum coefficient
        if not hasattr(self.args, 'ala_momentum') or self.args.ala_momentum == 0.0:
            self.args.ala_momentum = config["ala_momentum"]

        # Initialize ALA with temporal smoothing
        self.ala = ALA(self.client_id, self.task.loss_fn, self.task.splitted_data,
                       self.args.batch_size, self.args.rand_percent, self.args.layer_idx,
                       self.args.eta, self.device, self.args.threshold, self.args.num_pre_loss,
                       self.args.ala_beta)

        # Momentum buffer for update direction smoothing
        self.momentum_buffer = None
        self.momentum_gamma = self.args.ala_momentum

    def execute(self):
        """
        Executes the local training process:
        1. Copies global model parameters
        2. Computes momentum-smoothed update direction
        3. Applies ALA for adaptive layer-wise aggregation
        4. Trains locally
        """
        # Get global model
        global_model = copy.deepcopy(self.task.model)
        for (local_param, global_param) in zip(global_model.parameters(),
                                                self.message_pool["server"]["weight"]):
            local_param.data.copy_(global_param)

        # Initialize momentum buffer if first round
        if self.momentum_buffer is None:
            self.momentum_buffer = [torch.zeros_like(p.data) for p in self.task.model.parameters()]

        # Compute and apply momentum to update directions before ALA
        self._apply_momentum_update(global_model)

        # ALA: Adaptive Local Aggregation with temporal smoothing
        self.ala.adaptive_local_aggregation(global_model, self.task.model)

        # Train
        self.task.train()

    def _apply_momentum_update(self, global_model):
        """
        Apply momentum smoothing to the update direction.

        Updates momentum buffer: V_t = γ * V_(t-1) + (1-γ) * (global - local)
        Then modifies global_model params to incorporate momentum smoothing.
        """
        global_params = list(global_model.parameters())
        local_params = list(self.task.model.parameters())

        for i, (g_param, l_param) in enumerate(zip(global_params, local_params)):
            # Update direction: global - local
            update_dir = g_param.data - l_param.data

            # Momentum update: V_t = γ * V_(t-1) + (1-γ) * update_dir
            self.momentum_buffer[i] = (self.momentum_gamma * self.momentum_buffer[i].to(self.device) +
                                        (1 - self.momentum_gamma) * update_dir)

            # Apply momentum-smoothed direction to global model
            # This gives ALA a smoother target to aggregate towards
            g_param.data.copy_(l_param.data + self.momentum_buffer[i])

    def send_message(self):
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "weight": list(self.task.model.parameters())
        }
