import torch
import torch.nn as nn
import copy
from openfgl.flcore.base import BaseClient
from openfgl.flcore.fedala.ala import ALA
from openfgl.flcore.fedala_complete.fedala_complete_config import config


class FedALACompleteClient(BaseClient):
    """
    FedALACompleteClient combines ALL improvements to the FedALA algorithm:

    1. Temporal Smoothing (beta=0.3): Stabilizes aggregation weights across rounds
       α(t) = (1-β) * α_new + β * α(t-1)

    2. Proximal Regularization (mu=0.01): Prevents excessive drift from global model
       L_total = L_local + (μ/2) * ||W - W_global||²

    3. Momentum Integration (gamma=0.9): Smooths update direction over rounds
       V_t = γ * V_(t-1) + (1-γ) * (Θ_global - Θ_local)

    4. Meta-Learning Initialization: Preserves learned weights across rounds,
       reduces num_pre_loss from 10 to 3 after first round for faster convergence
    """

    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(FedALACompleteClient, self).__init__(args, client_id, data, data_dir, message_pool, device)

        # Set default FedALA args if not present
        if not hasattr(self.args, 'rand_percent'): self.args.rand_percent = 80
        if not hasattr(self.args, 'layer_idx'): self.args.layer_idx = 0
        if not hasattr(self.args, 'eta'): self.args.eta = 1.0
        if not hasattr(self.args, 'threshold'): self.args.threshold = 0.1
        if not hasattr(self.args, 'num_pre_loss'): self.args.num_pre_loss = 10

        # Temporal smoothing (Improvement 3.1)
        if not hasattr(self.args, 'ala_beta') or self.args.ala_beta == 0.0:
            self.args.ala_beta = config["ala_beta"]

        # Proximal regularization (Improvement 3.2)
        if not hasattr(self.args, 'ala_mu') or self.args.ala_mu == 0.0:
            self.args.ala_mu = config["ala_mu"]

        # Momentum (Improvement 3.3)
        if not hasattr(self.args, 'ala_momentum') or self.args.ala_momentum == 0.0:
            self.args.ala_momentum = config["ala_momentum"]

        # Meta-learning initialization (Improvement 3.4)
        if not hasattr(self.args, 'ala_meta_init'):
            self.args.ala_meta_init = config["ala_meta_init"]
        if not hasattr(self.args, 'ala_meta_init_epochs'):
            self.args.ala_meta_init_epochs = config["ala_meta_init_epochs"]

        # Initialize ALA with temporal smoothing
        self.ala = ALA(self.client_id, self.task.loss_fn, self.task.splitted_data,
                       self.args.batch_size, self.args.rand_percent, self.args.layer_idx,
                       self.args.eta, self.device, self.args.threshold, self.args.num_pre_loss,
                       self.args.ala_beta)

        # Momentum buffer for update direction smoothing
        self.momentum_buffer = None
        self.momentum_gamma = self.args.ala_momentum

        # Track round number for meta-learning initialization
        self.round_count = 0

    def get_custom_loss_fn(self):
        """
        Returns a custom loss function that includes the proximal regularization term.
        """
        def custom_loss_fn(embedding, logits, label, mask):
            # Proximal regularization term: (mu/2) * ||W_local - W_global||^2
            loss_prox = 0
            for local_param, global_param in zip(self.task.model.parameters(),
                                                  self.message_pool["server"]["weight"]):
                loss_prox += self.args.ala_mu / 2 * (local_param - global_param).norm(2)**2

            return self.task.default_loss_fn(logits[mask], label[mask]) + loss_prox

        return custom_loss_fn

    def execute(self):
        """
        Executes the local training process with ALL improvements:
        1. Copies global model parameters
        2. Applies momentum smoothing to update direction
        3. Applies ALA for adaptive layer-wise aggregation
        4. Applies meta-learning initialization (reduced epochs after first round)
        5. Trains with proximal regularization loss
        """
        self.round_count += 1

        # Get global model
        global_model = copy.deepcopy(self.task.model)
        for (local_param, global_param) in zip(global_model.parameters(),
                                                self.message_pool["server"]["weight"]):
            local_param.data.copy_(global_param)

        # Initialize momentum buffer if first round
        if self.momentum_buffer is None:
            self.momentum_buffer = [torch.zeros_like(p.data) for p in self.task.model.parameters()]

        # Improvement 3.3: Apply momentum to update directions before ALA
        self._apply_momentum_update(global_model)

        # Improvement 3.4: Meta-learning initialization
        # After first round, reduce num_pre_loss for faster convergence
        # since we preserve the learned weights
        if self.round_count > 1 and self.args.ala_meta_init:
            self.ala.num_pre_loss = self.args.ala_meta_init_epochs

        # ALA with temporal smoothing (Improvement 3.1): Adaptive Local Aggregation
        self.ala.adaptive_local_aggregation(global_model, self.task.model)

        # Improvement 3.2: Apply proximal regularization during training
        self.task.loss_fn = self.get_custom_loss_fn()

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
            g_param.data.copy_(l_param.data + self.momentum_buffer[i])

    def send_message(self):
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "weight": list(self.task.model.parameters())
        }
