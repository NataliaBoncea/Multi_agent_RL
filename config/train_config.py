import math


class TrainConfig:
    def __init__(self):
        self.WIDTH = 100
        self.HEIGHT = 100
        self.Nr = 3   # Number of agents 
        self.Nt = 15  # Number of targets 
        self.num_episodes = 1000 
        
        # Motion
        self.Vr = 1.0 # Search velocity 
        self.max_yaw = math.pi / 4 # Max yaw angle 
        self.delta_yaw_options = [0, math.radians(20), math.radians(-20)] # Heading adjustments
        
        # Sensors
        self.Rv = 14.0   # Detection radius 
        self.Rs = 1.0    # Safe distance 
        self.Da = 3.0    # Avoidance distance 
        self.fa = 0.8    # Collision avoidance factor 
        self.pd = 0.9    # Detection probability 
        
        # Rewards 
        self.r1 = 5.0    # Reward for finding a target 
        self.r2 = 100.0  # Reward for finding all targets 
        self.rp = -3.0   # Boundary penalty 
        self.rm = -1   # Movement cost (Tweaked for stability, originally -1.0) 
        self.rc = -5.0   # Collision penalty 
        self.rho = 5.0   # Presence penalty factor 
        # Reward per NEW pixel found. 
        # Example: 100 new pixels * 0.01 = 1.0 reward.
        self.r_cover_scale = 0.01 
        self.r_cover_penalty = -0.1 # Penalty if NO new area is seen (discourages spinning)
        
        # Algorithm Hyperparameters
        self.gamma = 0.95 # Discount factor 
        self.lr = 0.0005
        self.omega2 = 0.95 # Action preference weight 
        self.hidden_size = 64

        # Agents
        self.coll_alpha = 1.0
        self.coll_beta = 0.1
        
        # Steps
        self.max_steps = 400      # Max iteration steps 
        self.test_steps = 1000    
        
        # Epsilon Annealing
        self.epsilon_start = 0.5
        self.epsilon_end = 0.0
        self.epsilon_decay = (self.epsilon_start - self.epsilon_end) / self.num_episodes
        
        # Environment setting
        self.start_mode = "distributed"

        # System
        self.ckpt_dir = "checkpoints_coverage_dist"
        self.log_file = "training_log.csv"
        self.save_freq = 50

