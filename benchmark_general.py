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
        self.dataset = ["Cora"] # Will be overwritten
        self.simulation_mode = "subgraph_fl_louvain"
        self.num_clients = 10 # Will be overwritten
        self.num_rounds = 30 # Reduced for faster general benchmarking
        self.fl_algorithm = "fedavg" # Will be overwritten
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

# Configuration for Benchmark
algorithms_to_test = ["fedavg", "fedprox", "scaffold", "moon", "fedproto", "fedala", "fedala_s"]
client_counts_to_test = [5, 10, 20]
dataset_list = ["Cora", "CiteSeer", "PubMed", "Photo", "Computers", "Chameleon", "Actor", "Amazon-ratings"]

results = {} # Structure: results[dataset][num_clients][algo] = accuracy

print(f"Starting General Benchmark...")
print(f"Algorithms: {algorithms_to_test}")
print(f"Client Counts: {client_counts_to_test}")
print(f"Datasets: {dataset_list}")

for dataset in dataset_list:
    results[dataset] = {}
    print(f"\n{'='*30}\nBenchmark Dataset: {dataset}\n{'='*30}")
    
    for num_clients in client_counts_to_test:
        results[dataset][num_clients] = {}
        print(f"\n   >>> Client Count: {num_clients} <<<")
        
        for algo in algorithms_to_test:
            print(f"\n   Running {algo} with {num_clients} clients on {dataset}...")
            
            args = Args()
            args.dataset = [dataset]
            args.num_clients = num_clients
            args.fl_algorithm = algo
            
            # Model Selection Logic
            if dataset in ["Chameleon", "Actor", "Amazon-ratings"]:
                args.model = ["mlp"] # Simple heuristic for this script
            else:
                args.model = ["gcn"]

            # OOM Mitigation
            if dataset == "ogbn-products":
                args.use_cuda = False
                print(f"   [Info] Forcing CPU for {dataset} to avoid OOM.")
            else:
                args.use_cuda = torch.cuda.is_available()

            config.args = args
            
            try:
                trainer = FGLTrainer(args)
                trainer.train()
                
                acc = trainer.evaluation_result.get("best_test_accuracy", 0.0)
                print(f"   Result: {acc:.4f}")
                results[dataset][num_clients][algo] = acc
                
            except Exception as e:
                import traceback
                # traceback.print_exc()
                print(f"   Error: {e}")
                results[dataset][num_clients][algo] = "Error"

# Generate timestamp string
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"general_benchmark_results_{timestamp}.txt"

# Save and Print Results
print("\n\n=== Final General Benchmark Results ===")
with open(output_filename, "w") as f:
    header = f"{'Dataset':<15} | {'Clients':<8} | {'Algorithm':<15} | {'Test Accuracy':<15}\n"
    f.write(header)
    f.write("-" * 60 + "\n")
    print(header.strip())
    print("-" * 60)
    
    for dataset in dataset_list:
        for num_clients in client_counts_to_test:
            for algo in algorithms_to_test:
                acc = results[dataset][num_clients].get(algo, "N/A")
                if isinstance(acc, float):
                    line = f"{dataset:<15} | {num_clients:<8} | {algo:<15} | {acc:.4f}\n"
                else:
                    line = f"{dataset:<15} | {num_clients:<8} | {algo:<15} | {acc}\n"
                
                f.write(line)
                print(line.strip())
            f.write("-" * 60 + "\n")
            print("-" * 60)

print(f"\nFull results saved to {output_filename}")
