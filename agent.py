import math
import random


# --- AGENT ---
class Agent:
    def __init__(self, id, x, y, config):
        self.id = id
        self.x = x
        self.y = y
        self.phi = random.uniform(-math.pi, math.pi)
        self.config = config
        self.alpha = 1.0 # Alpha for Beta distribution 
        self.beta = 1.0  # Beta for Beta distribution 
        self.hidden_state = None 

    def move(self, action_idx, neighbors):
        delta_phi = self.config.delta_yaw_options[action_idx]

        # Safety: enforce paper turning limit
        if abs(delta_phi) > self.config.max_yaw:
            delta_phi = math.copysign(self.config.max_yaw, delta_phi)

        self.phi += delta_phi

        # Keep phi in [-pi, pi] for numerical stability
        self.phi = (self.phi + math.pi) % (2 * math.pi) - math.pi
        
        fx, fy = 0, 0
        collision_penalty = 0
        
        # Collision Avoidance Mechanism 
        for other in neighbors:
            dx = self.x - other.x
            dy = self.y - other.y
            dist = math.sqrt(dx**2 + dy**2)
            
            if self.config.Rs < dist < self.config.Da:
                # Incremental function for avoidance (Eq 3)
                factor = (self.config.Rs * self.config.fa * self.config.Vr) / (dist**2 + 1e-6)
                fx += factor * dx
                fy += factor * dy
            
            if dist <= self.config.Rs:
                dist_penalty = self.config.coll_alpha * dist
                hx, hy = math.cos(self.phi), math.sin(self.phi)
                Rx, Ry = 0.0, 1.0

                dot = hx * Rx + hy * Ry
                dot = max(-1.0, min(1.0, dot))

                angle = math.acos(dot)
                dir_penalty = (self.config.coll_beta / math.pi) * angle

                collision_penalty -= (dist_penalty + dir_penalty)

        
        # Update Position (Eq 4)
        next_x = self.x + self.config.Vr * math.cos(self.phi) + fx
        next_y = self.y + self.config.Vr * math.sin(self.phi) + fy
        
        boundary_penalty = 0
        if next_x <= 0 or next_x >= self.config.WIDTH or next_y <= 0 or next_y > self.config.HEIGHT:
            boundary_penalty = self.config.rp
            self.phi += math.pi/2 
            next_x = max(0, min(self.config.WIDTH, next_x))
            next_y = max(0, min(self.config.HEIGHT, next_y))
            
        self.x = next_x
        self.y = next_y
        return collision_penalty + boundary_penalty

    def get_odor_reward(self, active_targets):
        if not active_targets: return 0
        min_dist = min([math.sqrt((self.x - t[0])**2 + (self.y - t[1])**2) for t in active_targets])
        if 18 < min_dist <= 20: return 1
        elif 16 < min_dist <= 18: return 2
        elif 14 < min_dist <= 16: return 3
        return 0