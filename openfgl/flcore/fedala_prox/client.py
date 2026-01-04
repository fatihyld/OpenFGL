import torch
import torch.nn as nn
import copy
from openfgl.flcore.base import BaseClient
from openfgl.flcore.fedala.ala import ALA
from openfgl.flcore.fedala_prox.fedala_prox_config import config


class FedALAProxClient(BaseClient):
    """
    FedALAProxClient combines FedALA-S (Adaptive Local Aggregation with temporal smoothing)
    with FedProx (proximal regularization term).

    This implementation:
    1. Uses ALA for adaptive layer-wise aggregation between local and global models
    2. Applies temporal smoothing (beta) to stabilize aggregation weights
    3. Adds proximal regularization term to prevent local models from drifting too far

    The proximal term: L_prox = L_local + (mu/2) * ||W - W_global||^2
    """

    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(FedALAProxClient, self).__init__(args, client_id, data, data_dir, message_pool, device)

        # Set default FedALA args if not present
        if not hasattr(self.args, 'rand_percent'): self.args.rand_percent = 80
        if not hasattr(self.args, 'layer_idx'): self.args.layer_idx = 0
        if not hasattr(self.args, 'eta'): self.args.eta = 1.0
        if not hasattr(self.args, 'threshold'): self.args.threshold = 0.1
        if not hasattr(self.args, 'num_pre_loss'): self.args.num_pre_loss = 10

        # FedALA-S: temporal smoothing with beta
        if not hasattr(self.args, 'ala_beta') or self.args.ala_beta == 0.0:
            self.args.ala_beta = 0.3

        # FedALA-Prox: proximal regularization coefficient
        if not hasattr(self.args, 'ala_mu') or self.args.ala_mu == 0.0:
            self.args.ala_mu = config["ala_mu"]

        self.ala = ALA(self.client_id, self.task.loss_fn, self.task.splitted_data,
                       self.args.batch_size, self.args.rand_percent, self.args.layer_idx,
                       self.args.eta, self.device, self.args.threshold, self.args.num_pre_loss,
                       self.args.ala_beta)

    def get_custom_loss_fn(self):
        """
        Returns a custom loss function that includes the proximal regularization term.
        The proximal term penalizes deviation of local model parameters from global model.

        Returns:
            custom_loss_fn: Loss function with proximal term
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
        Executes the local training process:
        1. Copies global model parameters
        2. Applies ALA for adaptive layer-wise aggregation
        3. Trains with proximal regularization loss
        """
        # Get global model
        global_model = copy.deepcopy(self.task.model)
        for (local_param, global_param) in zip(global_model.parameters(),
                                                self.message_pool["server"]["weight"]):
            local_param.data.copy_(global_param)

        # ALA: Adaptive Local Aggregation with temporal smoothing
        self.ala.adaptive_local_aggregation(global_model, self.task.model)

        # Apply proximal regularization during training
        self.task.loss_fn = self.get_custom_loss_fn()

        # Train
        self.task.train()

    def send_message(self):
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "weight": list(self.task.model.parameters())
        }
