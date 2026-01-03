import torch
import torch.nn as nn
import copy
import random
import numpy as np

class ALA:
    def __init__(self, cid, loss, train_data, batch_size, rand_percent, layer_idx, eta, device, threshold, num_pre_loss, beta=0.0):
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

        self.weights = None # The learned weights
        self.start_phase = True

    def adaptive_local_aggregation(self, global_model, local_model):
        # Initialize weights if not exists
        if self.weights is None:
            self.weights = []
            for param in local_model.parameters():
                self.weights.append(torch.ones_like(param).to(self.device))

        # Prepare weights for optimization
        learned_weights = [w.clone().detach().requires_grad_(True) for w in self.weights]
        optimizer = torch.optim.Adam(learned_weights, lr=self.eta)

        # Temp model
        temp_model = copy.deepcopy(local_model)
        temp_model.eval()
        for param in temp_model.parameters():
            param.requires_grad = False

        # Data
        if isinstance(self.train_data, dict):
            data = self.train_data["data"]
            train_mask = self.train_data["train_mask"]
            train_indices = train_mask.nonzero().squeeze()
            num_samples = int(len(train_indices) * self.rand_percent / 100.0)
            if num_samples < 1: num_samples = 1
            sampled_indices = train_indices[torch.randperm(len(train_indices))[:num_samples]]
        else:
            # Fallback for other task types if needed
            return

        # Training loop
        for epoch in range(self.num_pre_loss):
            optimizer.zero_grad()
            
            params_global = list(global_model.parameters())
        # Pre-collect modules and names to access parameters
        params_structure = []
        for name, _ in temp_model.named_parameters():
            parts = name.split('.')
            module = temp_model
            for part in parts[:-1]:
                module = getattr(module, part)
            param_name = parts[-1]
            params_structure.append((module, param_name))

        # Training loop
        for epoch in range(self.num_pre_loss):
            optimizer.zero_grad()
            
            params_global = list(global_model.parameters())
            params_local = list(local_model.parameters())
            
            # Replace parameters with weighted combination
            for (module, param_name), w, p_g, p_l in zip(params_structure, learned_weights, params_global, params_local):
                # Calculate new weight
                # We assume p_g and p_l are fixed (requires_grad=False)
                new_val = w * p_g + (1 - w) * p_l
                
                # Replace in temp_model
                if hasattr(module, param_name):
                    delattr(module, param_name)
                setattr(module, param_name, new_val)
            
            # Forward pass
            # We assume the model forward signature matches NodeClsTask
            embedding, logits = temp_model(data)
            
            # Create mask for sampled indices
            mask = torch.zeros_like(train_mask).bool()
            mask[sampled_indices] = True
            
            loss = self.loss(embedding, logits, data.y, mask)
            
            loss.backward()
            optimizer.step()
            
        # Update weights
        with torch.no_grad():
            if self.beta > 0 and not self.start_phase:
                 # FedALA-S: Apply temporal smoothing
                 # alpha(t) = (1 - beta) * alpha_new + beta * alpha(t-1)
                 # alpha_new is learned_weights, alpha(t-1) is self.weights (before update)
                 
                 # DEBUG PRINT
                 # print(f"[ALA Debug] Client {self.cid}: Smoothing active. Beta={self.beta}. First weight old: {self.weights[0].flatten()[:3]}, new: {learned_weights[0].flatten()[:3]}")
                 
                 self.weights = [(1 - self.beta) * w_new + self.beta * w_old 
                                 for w_new, w_old in zip(learned_weights, self.weights)]
            else:
                 # if self.beta > 0:
                 #    print(f"[ALA Debug] Client {self.cid}: First round or start phase. Beta={self.beta}. No smoothing yet.")
                 self.weights = [w.clone() for w in learned_weights]
            
            self.start_phase = False
            
        # Update local model
        with torch.no_grad():
            for param_local, param_global, weight in zip(local_model.parameters(), global_model.parameters(), self.weights):
                param_local.data.copy_(weight * param_global + (1 - weight) * param_local)
