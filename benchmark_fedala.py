import openfgl.config as config
from openfgl.flcore.trainer import FGLTrainer
import sys
import torch

# Mock args class to allow dynamic modification
class Args:
    def __init__(self):
        self.root = "your_data_root"
        self.scenario = "subgraph_fl"
        self.dataset = ["Cora"] # Will be overwritten
        self.simulation_mode = "subgraph_fl_louvain"
        self.num_clients = 10
        self.num_rounds = 50 # Enough rounds for convergence
        self.fl_algorithm = "fedala"
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
        self.eta = 0.1
        self.threshold = 0.1
        self.num_pre_loss = 5

dataset_list = ["Cora", "CiteSeer", "PubMed"]
results = {}

for dataset in dataset_list:
    print(f"\nRunning benchmark for {dataset}...")
    args = Args()
    args.dataset = [dataset]
    config.args = args
    
    try:
        trainer = FGLTrainer(args)
        trainer.train()
        
        # The trainer saves results in trainer.evaluation_result
        best_acc = trainer.evaluation_result.get("best_test_accuracy", 0.0)
        results[dataset] = best_acc
        print(f"Result for {dataset}: {best_acc:.4f}")
    except Exception as e:
        print(f"Error running {dataset}: {e}")
        results[dataset] = "Error"

print("\n\n=== Final Results ===")
print(f"{'Dataset':<15} | {'Test Accuracy':<15}")
print("-" * 33)
for d, acc in results.items():
    val = f"{acc:.4f}" if isinstance(acc, float) else str(acc)
    print(f"{d:<15} | {val:<15}")
