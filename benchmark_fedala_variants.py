"""
Benchmark script for comparing FedALA variants.

Algorithms tested:
1. fedavg          - baseline
2. fedprox         - baseline with proximal term
3. fedala          - base FedALA
4. fedala_s        - with temporal smoothing (Improvement 3.1)
5. fedala_prox     - with smoothing + proximal (Improvement 3.2)
6. fedala_momentum - with smoothing + momentum (Improvement 3.3)
7. fedala_complete - all improvements combined (3.1 + 3.2 + 3.3 + 3.4)

Datasets: Cora, CiteSeer, PubMed, Photo, Computers, Chameleon, Actor, Amazon-ratings
Client configurations: N=5, 10, 20
"""

import openfgl.config as config
from openfgl.flcore.trainer import FGLTrainer
import sys
import torch
import torch_geometric.data.data
import torch_geometric.data.storage
import datetime
import gc

# --- Security Patch for PyTorch 2.6+ ---
try:
    torch.serialization.add_safe_globals([
        torch_geometric.data.data.DataEdgeAttr,
        torch_geometric.data.data.DataTensorAttr,
        torch_geometric.data.storage.GlobalStorage
    ])
except AttributeError:
    pass
# ---------------------------------------


class Args:
    def __init__(self):
        self.root = "your_data_root"
        self.scenario = "subgraph_fl"
        self.dataset = ["Cora"]
        self.simulation_mode = "subgraph_fl_louvain"
        self.num_clients = 10
        self.num_rounds = 30
        self.fl_algorithm = "fedavg"
        self.model = ["gcn"]
        self.metrics = ["accuracy"]
        self.use_cuda = torch.cuda.is_available()
        self.gpuid = 0
        self.seed = 2024
        self.processing = "raw"
        self.processing_percentage = 0.1
        self.feature_mask_prob = 0.1
        self.dp_epsilon = 0.0
        self.homo_injection_ratio = 0.0
        self.hete_injection_ratio = 0.0
        self.client_frac = 1.0
        self.dirichlet_alpha = 10
        self.dirichlet_try_cnt = 100
        self.least_samples = 5
        self.louvain_resolution = 1
        self.louvain_delta = 20
        self.metis_num_coms = 100
        self.task = "node_cls"
        self.num_clusters = 7
        self.train_val_test = "default_split"
        self.num_epochs = 3
        self.dropout = 0.5
        self.lr = 1e-2
        self.optim = "adam"
        self.weight_decay = 5e-4
        self.batch_size = 128
        self.num_layers = 2
        self.hid_dim = 64
        self.evaluation_mode = "local_model_on_local_data"
        self.dp_mech = "no_dp"
        self.noise_scale = 1.0
        self.grad_clip = 1.0
        self.dp_q = 0.1
        self.max_degree = 5
        self.max_epsilon = 20
        self.debug = False
        self.log_root = None
        self.log_name = None
        self.comm_cost = False
        self.model_param = False

        # Algorithm specific defaults
        self.mu = 0.01               # FedProx
        self.rand_percent = 80       # FedALA
        self.layer_idx = 0           # FedALA
        self.eta = 0.01              # FedALA
        self.threshold = 0.1         # FedALA
        self.num_pre_loss = 5        # FedALA
        self.ala_beta = 0.3          # FedALA-S: temporal smoothing
        self.ala_mu = 0.01           # FedALA-Prox: proximal coefficient
        self.ala_momentum = 0.9      # FedALA-Momentum: momentum coefficient
        self.ala_meta_init = True    # FedALA-Complete: meta-learning init
        self.ala_meta_init_epochs = 3  # FedALA-Complete: reduced epochs


# Benchmark Configuration
datasets_to_test = ["Cora", "CiteSeer", "PubMed", "Photo", "Computers", "Chameleon", "Actor", "Amazon-ratings"]
client_counts = [5, 10, 20]

# All FedALA variants + baselines
algorithms_to_test = [
    "fedavg",          # Baseline
    "fedprox",         # Baseline with proximal term
    "fedala",          # Base FedALA
    "fedala_s",        # + Temporal Smoothing (3.1)
    "fedala_prox",     # + Temporal Smoothing + Proximal (3.1 + 3.2)
    "fedala_momentum", # + Temporal Smoothing + Momentum (3.1 + 3.3)
    "fedala_complete"  # All improvements (3.1 + 3.2 + 3.3 + 3.4)
]

# Store all results
results = {}

# Calculate total experiments
total_experiments = len(datasets_to_test) * len(client_counts) * len(algorithms_to_test)
current_experiment = 0

print(f"\n{'='*70}")
print(f"FedALA Variants Benchmark")
print(f"{'='*70}")
print(f"Datasets: {datasets_to_test}")
print(f"Client counts: {client_counts}")
print(f"Algorithms: {algorithms_to_test}")
print(f"Total experiments: {total_experiments}")
print(f"{'='*70}\n")

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"fedala_variants_benchmark_{timestamp}.txt"

# Run all experiments
for dataset in datasets_to_test:
    for num_clients in client_counts:
        for algo in algorithms_to_test:
            current_experiment += 1
            key = (dataset, num_clients, algo)

            print(f"\n>>> [{current_experiment}/{total_experiments}] {dataset} | N={num_clients} | {algo}")

            args = Args()
            args.dataset = [dataset]
            args.num_clients = num_clients
            args.fl_algorithm = algo
            args.model = ["gcn"]

            config.args = args

            try:
                trainer = FGLTrainer(args)
                trainer.train()

                acc = trainer.evaluation_result.get("best_test_accuracy", 0.0)
                print(f"    Result: {acc:.4f}")
                results[key] = acc

            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"    Error: {e}")
                results[key] = "Error"

            # Clean up memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Save intermediate results after each dataset/client combination
        print(f"\n--- Saving intermediate results ---")
        with open(output_filename, "w") as f:
            f.write(f"FedALA Variants Benchmark Results\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"{'='*80}\n\n")

            f.write(f"{'Dataset':<15} | {'Clients':<8} | {'Algorithm':<17} | {'Test Accuracy':<15}\n")
            f.write("-" * 65 + "\n")

            for (ds, nc, alg), acc in sorted(results.items()):
                if isinstance(acc, float):
                    f.write(f"{ds:<15} | {nc:<8} | {alg:<17} | {acc:.4f}\n")
                else:
                    f.write(f"{ds:<15} | {nc:<8} | {alg:<17} | {acc}\n")

# Final Results Summary
print(f"\n\n{'='*80}")
print(f"FINAL RESULTS SUMMARY")
print(f"{'='*80}\n")

print(f"{'Dataset':<15} | {'Clients':<8} | {'Algorithm':<17} | {'Test Accuracy':<15}")
print("-" * 65)

for (dataset, num_clients, algo), acc in sorted(results.items()):
    if isinstance(acc, float):
        print(f"{dataset:<15} | {num_clients:<8} | {algo:<17} | {acc:.4f}")
    else:
        print(f"{dataset:<15} | {num_clients:<8} | {algo:<17} | {acc}")

# Save final results
with open(output_filename, "w") as f:
    f.write(f"FedALA Variants Benchmark Results\n")
    f.write(f"Timestamp: {timestamp}\n")
    f.write(f"{'='*80}\n\n")

    f.write(f"{'Dataset':<15} | {'Clients':<8} | {'Algorithm':<17} | {'Test Accuracy':<15}\n")
    f.write("-" * 65 + "\n")

    for (ds, nc, alg), acc in sorted(results.items()):
        if isinstance(acc, float):
            f.write(f"{ds:<15} | {nc:<8} | {alg:<17} | {acc:.4f}\n")
        else:
            f.write(f"{ds:<15} | {nc:<8} | {alg:<17} | {acc}\n")

    # Add comparison summary per dataset
    f.write(f"\n\n{'='*80}\n")
    f.write("ALGORITHM COMPARISON BY DATASET (Average across client counts)\n")
    f.write(f"{'='*80}\n\n")

    for dataset in datasets_to_test:
        f.write(f"\n{dataset}:\n")
        f.write("-" * 40 + "\n")
        for algo in algorithms_to_test:
            accs = [results.get((dataset, nc, algo), 0) for nc in client_counts]
            valid_accs = [a for a in accs if isinstance(a, float)]
            if valid_accs:
                avg = sum(valid_accs) / len(valid_accs)
                f.write(f"  {algo:<17}: {avg:.4f} (N=5: {accs[0]:.4f if isinstance(accs[0], float) else 'Error'}, N=10: {accs[1]:.4f if isinstance(accs[1], float) else 'Error'}, N=20: {accs[2]:.4f if isinstance(accs[2], float) else 'Error'})\n")
            else:
                f.write(f"  {algo:<17}: Error\n")

print(f"\nResults saved to {output_filename}")
