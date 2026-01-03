import torch
import torch.nn as nn
import copy
from openfgl.flcore.base import BaseClient
from openfgl.flcore.fedala.ala import ALA

class FedALASClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(FedALASClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
        
        # Set default args if not present
        if not hasattr(self.args, 'rand_percent'): self.args.rand_percent = 80
        if not hasattr(self.args, 'layer_idx'): self.args.layer_idx = 0
        if not hasattr(self.args, 'eta'): self.args.eta = 1.0
        if not hasattr(self.args, 'threshold'): self.args.threshold = 0.1
        if not hasattr(self.args, 'num_pre_loss'): self.args.num_pre_loss = 10
        
        # FedALA-S specific: Default beta to 0.8 if not provided or 0
        if not hasattr(self.args, 'ala_beta') or self.args.ala_beta == 0.0:
            self.args.ala_beta = 0.3

        self.ala = ALA(self.client_id, self.task.loss_fn, self.task.splitted_data, 
                       self.args.batch_size, self.args.rand_percent, self.args.layer_idx, 
                       self.args.eta, self.device, self.args.threshold, self.args.num_pre_loss,
                       self.args.ala_beta)

    def execute(self):
        # Get global model
        global_model = copy.deepcopy(self.task.model)
        for (local_param, global_param) in zip(global_model.parameters(), self.message_pool["server"]["weight"]):
            local_param.data.copy_(global_param)

        # ALA
        self.ala.adaptive_local_aggregation(global_model, self.task.model)

        # Train
        self.task.train()

    def send_message(self):
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "weight": list(self.task.model.parameters())
        }
