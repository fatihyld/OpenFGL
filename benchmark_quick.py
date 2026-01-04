import openfgl.config as config
from openfgl.flcore.trainer import FGLTrainer
import sys
import torch
import torch_geometric.data.data
import torch_geometric.data.storage
import datetime

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
        self.mu = 0.01 # FedProx
        self.rand_percent = 80 # FedALA
        self.layer_idx = 0 # FedALA
        self.eta = 0.01 # FedALA
        self.threshold = 0.1 # FedALA
        self.num_pre_loss = 5 # FedALA
        self.ala_beta = 0.8 # FedALA-S

# Quick Benchmark Configuration - Cora only with 10 clients
algorithms_to_test = ["fedavg", "fedprox", "scaffold", "moon", "fedproto", "fedala", "fedala_s"]
num_clients = 10
dataset = "Cora"

results = {}

print(f"\n{'='*60}")
print(f"Quick Benchmark: {dataset} with {num_clients} clients")
print(f"Algorithms: {algorithms_to_test}")
print(f"{'='*60}\n")

for algo in algorithms_to_test:
    print(f"\n>>> Running {algo}...")

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
        print(f"    {algo}: {acc:.4f}")
        results[algo] = acc

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"    Error: {e}")
        results[algo] = "Error"

# Save and Print Results
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"quick_benchmark_results_{timestamp}.txt"

print(f"\n\n{'='*60}")
print(f"RESULTS: {dataset} with {num_clients} clients")
print(f"{'='*60}")

with open(output_filename, "w") as f:
    header = f"Quick Benchmark Results - {dataset} - {num_clients} clients\n"
    header += f"Timestamp: {timestamp}\n"
    header += "="*40 + "\n\n"
    f.write(header)

    print(f"\n{'Algorithm':<15} | {'Test Accuracy':<15}")
    print("-" * 35)
    f.write(f"{'Algorithm':<15} | {'Test Accuracy':<15}\n")
    f.write("-" * 35 + "\n")

    for algo in algorithms_to_test:
        acc = results.get(algo, "N/A")
        if isinstance(acc, float):
            line = f"{algo:<15} | {acc:.4f}"
        else:
            line = f"{algo:<15} | {acc}"
        print(line)
        f.write(line + "\n")

print(f"\nResults saved to {output_filename}")
