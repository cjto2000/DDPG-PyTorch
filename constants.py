MAX_BUFF_SIZE = 300000
N_EPISODES = 50000
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 0.001
LOG_STEPS = 5
SAVE_STEPS = 20
THRESHOLD_STEPS = 1000
A_LEARNING_RATE = 1e-4
C_LEARNING_RATE = 1e-3

actor_model_path = "saved_models/actor.pth"
critic_model_path = "saved_models/critic.pth"

# SAVE STEPS PATHS
actor_save_path = "save_step_models/actor.pth"
critic_save_path = "save_step_models/critic.pth"
target_actor_save_path = "save_step_models/target_actor.pth"
target_critic_save_path = "save_step_models/target_critic.pth"
