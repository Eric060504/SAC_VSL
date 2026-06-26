"""
config.py — Central configuration for SAC-based VSL control on highway work zone.

All parameters for SUMO paths, network geometry, RL agent, training, and reward.
"""

import os

# ============================================================
# Paths
# ============================================================
SUMO_HOME = "D:/sumo-win64-1.26.0/sumo-1.26.0"
SUMO_BINARY = "sumo"                 # headless; use "sumo-gui" for visualization
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Python executable for the LLM_Classification conda env
PYTHON_EXE = "D:/anaconda/envs/LLM_Classification/python.exe"

# ============================================================
# Scenario Configs: name → (sumocfg_file, cav_penetration_rate)
# ============================================================
SCENARIOS = {
    "cav25":  ("speedcontrol9.sumocfg",  0.25),
    "cav50":  ("speedcontrol10.sumocfg", 0.50),
    "cav75":  ("speedcontrol1.sumocfg",  0.75),
    "cav100": ("speedcontrol2.sumocfg",  1.00),
}

# ============================================================
# Network Parameters (from net.net.xml)
# ============================================================
EDGES_ALL  = ["E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7"]
CONTROL_EDGES   = ["E1", "E2", "E3"]             # VSL applied to CAVs here
OBS_EDGES       = ["E0", "E1", "E2", "E3", "E4", "E5"]   # observation region
REWARD_EDGES    = ["E1", "E2", "E3", "E4", "E5"]  # reward calculation region
UPSTREAM_EDGE   = "E0"

LANES_PER_EDGE  = 3
LANE_INDICES    = [0, 1, 2]

# Edge lengths (meters)
EDGE_LENGTHS = {
    "E0": 500, "E1": 500, "E2": 500, "E3": 500,
    "E4": 200, "E5": 300, "E6":  50, "E7": 500,
}
# Edge speed limits (m/s) — from net.xml
EDGE_SPEED_LIMITS = {
    "E0": 33.30, "E1": 22.20, "E2": 22.20, "E3": 22.20,
    "E4": 22.20, "E5": 22.20, "E6": 22.20, "E7": 33.33,
}

# Vehicle type IDs for CAVs (consistent across all route files)
CAV_TYPES = {"CAV", "CAV_truck"}

# Vehicle effective length + min Gap for density estimation (meters)
VEH_EFFECTIVE_LENGTH = 7.5

# ============================================================
# Simulation Settings
# ============================================================
SIM_BEGIN       = 0
SIM_END         = 3600         # 1 hour (reduced from 3h for faster training)
SUMO_STEP_LENGTH = 1.0         # seconds per SUMO step (coarser for speed)
CONTROL_INTERVAL = 120         # seconds between RL decisions
STEPS_PER_CONTROL = int(CONTROL_INTERVAL / SUMO_STEP_LENGTH)  # 120

TOTAL_SIM_STEPS = int((SIM_END - SIM_BEGIN) / SUMO_STEP_LENGTH)  # 3600
CONTROL_STEPS_PER_EPISODE = TOTAL_SIM_STEPS // STEPS_PER_CONTROL   # 30

WARMUP_CONTROL_STEPS = 5       # first 600s (5×120s=600s): no control, no reward
TRAFFIC_FLOW_RATE = 3400       # veh/h total
TRUCK_RATIO = 0.20

# ============================================================
# VSL Action Bounds
# ============================================================
VSL_MIN = 8.33   # m/s (30 km/h) — minimum safe speed in work zone
VSL_MAX = 22.22  # m/s (80 km/h) — posted work zone speed limit
VSL_SMOOTHING_ALPHA = 0.3  # EMA smoothing factor

# ============================================================
# State / Action Dimensions
# ============================================================
STATE_DIM  = 44
ACTION_DIM = 3    # one VSL value per control edge (E1, E2, E3)

# ============================================================
# SAC Hyperparameters
# ============================================================
ACTOR_LR    = 3e-4
CRITIC_LR   = 3e-4
ALPHA_LR    = 3e-4
GAMMA       = 0.99
TAU         = 0.005          # soft target update coefficient
BUFFER_SIZE = 1_000_000
BATCH_SIZE  = 256
MIN_BUFFER_SIZE = 128        # start training after this many transitions (lower for faster start)
HIDDEN_SIZES = [256, 256, 128]
INITIAL_LOG_ALPHA = -2.3026  # ln(0.1)
TARGET_ENTROPY = -float(ACTION_DIM)  # -3.0 in SAC paper
GRADIENT_CLIP  = 1.0
UPDATES_PER_STEP = 5          # multiple gradient steps per env step (faster learning)

# ============================================================
# Training Settings
# ============================================================
TRAIN_EPISODES  = 300
EVAL_EPISODES   = 10
SAVE_INTERVAL   = 50
LOG_INTERVAL    = 1
CHECKPOINT_DIR  = os.path.join(PROJECT_DIR, "checkpoints")

# ============================================================
# Reward Weights
# ============================================================
W_SAFETY     = 0.35
W_EFFICIENCY = 0.35
W_COMFORT    = 0.15
W_THROUGHPUT = 0.15

# Safety sub-parameters
TTC_THRESHOLD     = 3.0    # seconds — TTC below this is critical
TTC_LOOKAHEAD     = 100.0  # meters — max distance to look for leader

# Efficiency sub-parameters
# Free-flow travel time E1→E5: (500+500+500+200+300) / 22.22 ≈ 90.0 s
FREEFLOW_TRAVEL_TIME = sum(EDGE_LENGTHS[e] for e in REWARD_EDGES) / EDGE_SPEED_LIMITS["E1"]

# Comfort sub-parameters
JERK_REFERENCE = 5.0   # m/s³ — reference jerk for exponential decay

# ============================================================
# E3 Detector Settings
# ============================================================
E3_DETECTOR_ID = "e3_tt_e1_e5"

# ============================================================
# Random Seeds
# ============================================================
SCENARIO_SEEDS = {
    "cav25":  0,
    "cav50":  1,
    "cav75":  2,
    "cav100": 3,
}

# ============================================================
# Additional SUMO options
# ============================================================
SUMO_ADDITIONAL_OPTS = [
    "--no-warnings",
    "--no-step-log",
    "--time-to-teleport", "-1",     # disable teleportation
    "--collision.action", "none",   # ignore collisions (RL training safety)
]
