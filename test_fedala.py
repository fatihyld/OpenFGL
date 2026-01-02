import openfgl.config as config
from openfgl.flcore.trainer import FGLTrainer
import sys

# Mock args
class Args:
    def __init__(self):
        self.root = "your_data_root"
        self.scenario = "subgraph_fl"
        self.dataset = ["Cora"]
        self.simulation_mode = "subgraph_fl_louvain"
        self.num_clients = 2
        self.num_rounds = 2
        self.fl_algorithm = "fedala"
        self.model = ["gcn"]
        self.metrics = ["accuracy"]
        self.use_cuda = False
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
        self.num_epochs = 1
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
        
        # ALA specific
        self.rand_percent = 80
        self.layer_idx = 0
        self.eta = 1.0
        self.threshold = 0.1
        self.num_pre_loss = 2 # Small number for testing

args = Args()
config.args = args

print("Initializing Trainer with FedALA...")
trainer = FGLTrainer(args)
print("Starting Training...")
trainer.train()
print("Training Completed.")
