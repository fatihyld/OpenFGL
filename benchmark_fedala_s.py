import openfgl.config as config
from openfgl.flcore.trainer import FGLTrainer
import sys
import torch
import torch_geometric.data.data
import torch_geometric.data.storage
try:
    torch.serialization.add_safe_globals([
        torch_geometric.data.data.DataEdgeAttr,
        torch_geometric.data.data.DataTensorAttr,
        torch_geometric.data.storage.GlobalStorage
    ])
except AttributeError:
    pass

import datetime



# Mock args class to allow dynamic modification
class Args:
    def __init__(self):
        self.root = "your_data_root"
        self.scenario = "subgraph_fl"
        self.dataset = ["Cora"] # Will be overwritten
        self.simulation_mode = "subgraph_fl_louvain"
        self.num_clients = config.args.num_clients
        self.num_rounds = 50 # Enough rounds for convergence
        self.fl_algorithm = "fedala_s" # FedALA-S mode
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
        
        # FedALA specific
        self.rand_percent = 80
        self.layer_idx = 0
        self.eta = 0.01
        self.threshold = 0.1
        self.num_pre_loss = 5

# "ogbn-products" # Removed because it causes Out Of Memory errors
dataset_list = ["Cora", "CiteSeer", "PubMed", "Photo", "Computers",  "Chameleon", "Actor", "Amazon-ratings"] 
results = {}

for dataset in dataset_list:
    print(f"\n{'='*20}\nBenchmark: {dataset} (FedALA-S)\n{'='*20}")
    
    # Select candidate models based on dataset characteristics
    # Heterophilic datasets often benefit from MLP, GAT, or SGC
    if dataset in ["Chameleon", "Actor", "Amazon-ratings"]:
        candidate_models = ["gcn", "gat", "mlp", "sgc"]
    else:
        candidate_models = ["gcn"]

    best_acc = 0.0
    best_model = ""

    for model_name in candidate_models:
        print(f"\n--- Testing model: {model_name} on {dataset} ---")
        
        args = Args()
        args.dataset = [dataset]
        args.model = [model_name]
        if dataset == "ogbn-products":
            args.use_cuda = False
            print(f"Forcing CPU for {dataset} to avoid OOM.")
        else:
            args.use_cuda = torch.cuda.is_available()
            
        config.args = args
        
        try:
            trainer = FGLTrainer(args)
            trainer.train()
            
            acc = trainer.evaluation_result.get("best_test_accuracy", 0.0)
            print(f"Result for {dataset} with {model_name}: {acc:.4f}")
            
            if isinstance(acc, float) and acc > best_acc:
                best_acc = acc
                best_model = model_name
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error running {dataset} with {model_name}: {e}")
            if best_acc == 0.0: # Only mark as error if no model succeeded yet
                best_acc = "Error"

    results[dataset] = f"{best_acc} ({best_model})" if isinstance(best_acc, float) else best_acc


# Generate timestamp string
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"fedala_s_results_{timestamp}.txt"

print("\n\n=== Final Results (FedALA-S) ===")
print_output = f"{'Dataset':<15} | {'Test Accuracy':<15}\n"
print_output += "-"*33 + "\n"
for dataset, acc in results.items():
    if isinstance(acc, float):
        print_output += f"{dataset:<15} | {acc:.4f}\n"
    else:
        print_output += f"{dataset:<15} | {acc}\n"

print(print_output)

# Save to file
with open(output_filename, "w") as f:
    f.write(print_output)
    f.write("\nDetailed Results Dictionary:\n")
    f.write(str(results))

print(f"\nResults saved to {output_filename}")
