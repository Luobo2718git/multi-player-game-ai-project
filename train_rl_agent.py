import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn
import gymnasium as gym

# Import environment and opponent AI
from games.bomb.bomb_env import BombEnv
from games.bomb.bomb_game import BombGame # Import BombGame to use its constants
from agents import BombAI # Your opponent AI

# Define a custom feature extractor to handle Dict observation space
# This will concatenate image and scalar features
class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # Initialize BaseFeaturesExtractor. The features_dim here is the *output* dimension of this extractor.
        # We will set it to 256, as that's the output size of our final linear layer.
        super().__init__(observation_space, features_dim=256)

        self.observation_space = observation_space

        # Image feature extractor (CNN)
        # Assuming image_observation_space is (channels, height, width)
        n_channels = observation_space["board_features"].shape[0]
        board_height = observation_space["board_features"].shape[1]
        board_width = observation_space["board_features"].shape[2]

        # Simple CNN architecture
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # Fixed output size to 64 x 4 x 4
            nn.Flatten()
        )

        # Calculate the flattened dimension of the CNN output
        with th.no_grad():
            # Create a dummy input tensor matching the expected shape (batch_size, channels, height, width)
            # We use 1 for batch_size as we only need to infer the feature dimension.
            dummy_input = th.as_tensor(np.zeros((1, n_channels, board_height, board_width)), dtype=th.float32)
            cnn_output_flattened_dim = self.cnn(dummy_input).shape[1]

        # Dimension of scalar features
        scalar_feature_dim = observation_space["scalar_features"].shape[0]

        # Total dimension before the final linear layer (CNN output + scalar features)
        combined_pre_linear_dim = cnn_output_flattened_dim + scalar_feature_dim

        # Final linear layer to combine features and output a fixed size (256)
        self.final_linear_layer = nn.Sequential(
            nn.Linear(combined_pre_linear_dim, 256), # Output 256 features
            nn.ReLU()
        )

        # The _features_dim attribute (inherited from BaseFeaturesExtractor) must reflect the *actual output*
        # dimension of this feature extractor's forward pass. This fixes the "mat1 and mat2 shapes" error.
        self._features_dim = 256

    def forward(self, observations: dict[str, th.Tensor]) -> th.Tensor:
        # Normalize image features: divide by BombGame.WILL_EXPLOSION (max value of board elements)
        board_features = observations["board_features"].float() / BombGame.WILL_EXPLOSION
        
        # Normalize scalar features: divide by the maximum possible value in the scalar observation space.
        scalar_high = self.observation_space["scalar_features"].high
        # Safely get the max value from the high attribute. If it's an array, take the max; otherwise, use it directly.
        scalar_max_val = scalar_high.max() if isinstance(scalar_high, np.ndarray) else scalar_high
        
        scalar_features = observations["scalar_features"].float() / scalar_max_val

        cnn_out = self.cnn(board_features)
        
        # Concatenate CNN output and scalar features along dimension 1 (features dimension)
        combined_features = th.cat((cnn_out, scalar_features), dim=1)
        
        # Pass through the final linear layer
        return self.final_linear_layer(combined_features)


def make_env(rank: int, seed: int = 0, player_id: int = 1, opponent_agent=None):
    """
    Function for creating environments for SubprocVecEnv
    """
    def _init():
        env = BombEnv(board_size=15, player_id=player_id, opponent_agent=opponent_agent)
        env.reset(seed=seed + rank)
        print(f"DEBUG: Initialized environment for rank {rank} with seed {seed + rank}")
        return env
    return _init

if __name__ == '__main__':
    LOG_DIR = "./ppo_bomb_logs/"
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs("./models/", exist_ok=True) # For saving checkpoints
    os.makedirs("./best_model/", exist_ok=True) # For saving the best model
    os.makedirs("./results/", exist_ok=True) # For saving evaluation results logs

    print("DEBUG: Directories created.")

    # Instantiate your opponent AI
    opponent_ai = BombAI(name="OpponentBombAI", player_id=2) # RL agent as Player 1, AI as Player 2
    print("DEBUG: Opponent AI instantiated.")

    # Create multiple parallel environments for training
    num_envs = 8 # Adjust based on your CPU cores
    try:
        vec_env = SubprocVecEnv([make_env(i, player_id=1, opponent_agent=opponent_ai) for i in range(num_envs)])
        print(f"DEBUG: SubprocVecEnv created with {num_envs} environments.")
    except Exception as e:
        print(f"ERROR: Failed to create SubprocVecEnv: {e}")
        # Fallback to DummyVecEnv for debugging if SubprocVecEnv fails
        print("INFO: Falling back to DummyVecEnv for single-process debugging.")
        vec_env = DummyVecEnv([make_env(0, player_id=1, opponent_agent=opponent_ai)])


    # Define policy network architecture for PPO model
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        # net_arch defines the layers *after* the feature extractor.
        # The first layer will take the output of CustomCombinedExtractor (256 features) as input.
        net_arch=dict(pi=[256, 128], vf=[256, 128]) 
    )
    print("DEBUG: Policy kwargs defined.")

    # Create PPO model
    model = PPO(
        "MultiInputPolicy", # For Dict observation space
        vec_env,
        policy_kwargs=policy_kwargs,
        verbose=1, # Print training information
        tensorboard_log=LOG_DIR, # TensorBoard log directory
        gamma=0.99, # Discount factor, larger value emphasizes long-term rewards
        n_steps=2048, # Number of steps to collect data before a learning update (per environment)
        ent_coef=0.01, # Entropy regularization coefficient, encourages exploration (0.01 is common)
        learning_rate=0.0003, # Learning rate
        clip_range=0.2, # PPO's clipping range
        batch_size=64, # Batch size used for each update
        gae_lambda=0.95 # Lambda parameter for Generalized Advantage Estimation (GAE)
    )
    print("DEBUG: PPO model created.")

    # Set up callbacks: save best model and periodic checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=100000, # Save model every 100,000 steps
        save_path="./models/",
        name_prefix="ppo_bomb_model"
    )
    print("DEBUG: Checkpoint callback set up.")

    # Create a separate evaluation environment (not used for training, only for evaluation)
    eval_env = DummyVecEnv([make_env(0, player_id=1, opponent_agent=opponent_ai)])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model/",
        log_path="./results/",
        eval_freq=50000, # Evaluate every 50,000 steps
        deterministic=True, # Use deterministic policy during evaluation
        render=False # Do not render during evaluation
    )
    print("DEBUG: Eval callback set up.")

    print("Starting PPO agent training...")
    total_timesteps = 5_000_000 # Total training steps, can be adjusted, 5 million steps is a starting point
    try:
        model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, eval_callback])
        print("Training complete!")
    except Exception as e:
        print(f"ERROR: Training failed: {e}")

    # Save the final model
    model.save("ppo_bomb_final_model")
    print("DEBUG: Final model saved.")

    # Test the trained model (optional)
    print("Testing the trained model...")
    # obs here is a dictionary of NumPy arrays, already batched by DummyVecEnv (batch_size=1)
    obs, info = eval_env.reset() 

    for episode in range(3): # Run a few episodes for testing
        done = False
        truncated = False
        step_count = 0
        print(f"DEBUG: Starting test episode {episode + 1}")
        while not (done or truncated):
            # model.predict expects a batched observation. DummyVecEnv already provides it.
            action, _states = model.predict(obs, deterministic=True)
            
            # Action returned by model.predict is batched (e.g., [action_value])
            # Unbatch it for the environment step as the single environment expects a single action.
            single_action = action[0] 

            obs, reward, done, truncated, info = eval_env.step(single_action)
            
            step_count += 1
            if done[0] or truncated[0]: # Accessing first element for vectorized env
                print(f"DEBUG: Episode {episode + 1} finished.")
                print(f"Episode ended, winner: {info[0].get('winner', 'N/A')}, Player 1 score: {info[0].get('player1_score', 'N/A')}, Player 2 score: {info[0].get('player2_score', 'N/A')}")
                break # Exit inner while loop

        if not (done[0] or truncated[0]): # If loop finished without done/truncated (e.g., max steps)
            print(f"DEBUG: Episode {episode + 1} reached max steps without ending.")
            print(f"Episode ended, winner: {info[0].get('winner', 'N/A')}, Player 1 score: {info[0].get('player1_score', 'N/A')}, Player 2 score: {info[0].get('player2_score', 'N/A')}")
        
        obs, info = eval_env.reset() # Reset for next episode

    eval_env.close()
    print("DEBUG: Eval environment closed.")
