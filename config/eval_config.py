import math


class EvalConfig:
    def __init__(self):
        self.WIDTH = 100
        self.HEIGHT = 100
        self.Nr = 3   
        self.Nt = 15  
        
        # Test Settings
        self.num_test_episodes = 100 # Increased for better statistical average
        self.max_steps = 500        # Standard max steps (matches training usually)
        
        # Motion & Sensors 
        self.Vr = 1.0 
        self.max_yaw = math.pi / 4 
        self.delta_yaw_options = [0, math.radians(20), math.radians(-20)]
        self.Rv = 14.0  
        self.Rs = 1.0    
        self.Da = 3.0    
        self.fa = 0.8    
        self.pd = 0.9    
        
        # Rewards (Needed for "Average Reward" metric)
        self.r1 = 5.0    # Found target
        self.r2 = 100.0  # All found bonus
        self.rp = -3.0   # Boundary
        self.rm = -1     # Movement cost
        self.rc = -5.0   # Collision
        self.rho = 5.0   # Presence/Time penalty factor
        
        self.hidden_size = 64 
        self.ckpt_dir = "checkpoint_coverage"