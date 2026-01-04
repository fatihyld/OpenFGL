"""
Quick test to verify new FedALA variant implementations.
Tests: fedala_prox, fedala_momentum, fedala_complete
"""

import openfgl.config as config
from openfgl.flcore.trainer import FGLTrainer
import torch
import torch_geometric.data.data
import torch_geometric.data.storage

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
        self.num_clients = 5
        self.num_rounds = 10  # Reduced for quick test
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
        self.mu = 0.01
        self.rand_percent = 80
        self.layer_idx = 0
        self.eta = 0.01
        self.threshold = 0.1
        self.num_pre_loss = 5
        self.ala_beta = 0.3
        self.ala_mu = 0.01
        self.ala_momentum = 0.9
        self.ala_meta_init = True
        self.ala_meta_init_epochs = 3


# Test new algorithms
new_algorithms = ["fedala_prox", "fedala_momentum", "fedala_complete"]
results = {}

print("="*60)
print("Quick Test: Verifying New FedALA Variants")
print("Dataset: Cora | Clients: 5 | Rounds: 10")
print("="*60)

for algo in new_algorithms:
    print(f"\n>>> Testing {algo}...")

    args = Args()
    args.fl_algorithm = algo
    config.args = args

    try:
        trainer = FGLTrainer(args)
        trainer.train()

        acc = trainer.evaluation_result.get("best_test_accuracy", 0.0)
        print(f"    SUCCESS: {acc:.4f}")
        results[algo] = acc

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"    FAILED: {e}")
        results[algo] = "Error"

print("\n" + "="*60)
print("TEST RESULTS")
print("="*60)

all_passed = True
for algo in new_algorithms:
    acc = results.get(algo, "N/A")
    status = "PASS" if isinstance(acc, float) else "FAIL"
    if status == "FAIL":
        all_passed = False
    print(f"{algo:<18}: {status} ({acc})")

print("\n" + "="*60)
if all_passed:
    print("ALL TESTS PASSED - Ready for full benchmark")
else:
    print("SOME TESTS FAILED - Check implementation")
print("="*60)
