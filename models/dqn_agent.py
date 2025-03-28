# Generated with write_code_to_file
# /home/myuser/apps/battleshiprl/models/dqn_agent.py

import numpy as np
import random
import os
import logging
from typing import Tuple, List, Optional, Dict, Any
from collections import deque, namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Assuming the game constants are defined somewhere accessible
# If not, define or import them explicitly. Example:
try:
    # Adjust the import path based on your project structure
    from battleship.game import BOARD_SIZE, BOARD_EMPTY, BOARD_HIT, BOARD_MISS
except ImportError:
    logging.warning("Could not import game constants from battleship.game. Using default values.")
    BOARD_SIZE = 10
    BOARD_EMPTY = 0
    BOARD_HIT = 2
    BOARD_MISS = 3

# Set up logging
logger = logging.getLogger(__name__)
# Ensure logger has handlers (configure root logger in main.py or trainer.py)
# If run standalone, uncomment for basic logging:
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Define the structure for experiences stored in the replay buffer
Transition = namedtuple('Transition',
                        ('state', 'action_index', 'reward', 'next_state', 'done'))

# Define the neural network architecture for the Q-function approximation
class DQN(nn.Module):
    """Deep Q-Network for Battleship."""

    def __init__(self, input_size: int, output_size: int, hidden_size: int = 128):
        """
        Initializes the DQN model.

        Args:
            input_size (int): Size of the input state vector (e.g., board_size * board_size).
            output_size (int): Number of possible actions (e.g., board_size * board_size).
            hidden_size (int): Size of the hidden layers.
        """
        super(DQN, self).__init__()
        logger.debug(f"Initializing DQN with input={input_size}, output={output_size}, hidden={hidden_size}")
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size) # Output layer provides Q-values for each action

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor representing the game state.
                              Expected shape: (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor representing Q-values for each action.
                          Shape: (batch_size, output_size).
        """
        # Ensure input is flattened
        x = x.view(x.size(0), -1) # Flatten input if it's not already
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        q_values = self.fc3(x)
        return q_values

# Replay Memory Buffer
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, capacity: int):
        """
        Initializes the ReplayBuffer.

        Args:
            capacity (int): Maximum number of transitions to store.
        """
        logger.info(f"Initializing ReplayBuffer with capacity: {capacity}")
        self.memory = deque([], maxlen=capacity)
        self.capacity = capacity

    def push(self, *args: Any):
        """Saves a transition `(state, action_index, reward, next_state, done)`."""
        # Ensure states are converted to a suitable format (e.g., tensors) if needed,
        # although sampling is where it's more critical.
        self.memory.append(Transition(*args))
        # logger.debug(f"Pushed transition. Buffer size: {len(self.memory)}/{self.capacity}")

    def sample(self, batch_size: int) -> Optional[List[Transition]]:
        """
        Samples a random batch of transitions from memory.

        Args:
            batch_size (int): The number of transitions to sample.

        Returns:
            Optional[List[Transition]]: A list of sampled transitions, or None if not enough samples.
        """
        if len(self.memory) < batch_size:
            logger.debug(f"Not enough samples in buffer ({len(self.memory)}) to provide batch size {batch_size}.")
            return None
        sampled_transitions = random.sample(self.memory, batch_size)
        logger.debug(f"Sampled {batch_size} transitions from buffer.")
        return sampled_transitions

    def __len__(self) -> int:
        """Return the current size of internal memory."""
        return len(self.memory)


class DQNAgent:
    """Deep Q-Learning Agent for playing Battleship."""

    def __init__(self,
                 board_size: int = BOARD_SIZE,
                 learning_rate: float = 0.0005, # Adjusted typical LR for DQN
                 gamma: float = 0.99,           # Discount factor
                 epsilon_start: float = 1.0,    # Initial exploration rate
                 epsilon_decay: float = 0.9995, # Decay rate for exploration
                 epsilon_min: float = 0.05,     # Minimum exploration rate
                 buffer_size: int = 10000,      # Replay buffer capacity
                 batch_size: int = 64,          # Training batch size
                 target_update_freq: int = 10, # How often to update target network (in #learn calls)
                 tau: float = 0.005,            # Soft update parameter (if used, set target_update_freq=1)
                 use_soft_update: bool = True, # Whether to use soft or hard target updates
                 model_path: Optional[str] = None,
                 device: Optional[torch.device] = None):
        """
        Initializes the DQN agent.

        Args:
            board_size (int): Dimension of the game board.
            learning_rate (float): Learning rate for the optimizer.
            gamma (float): Discount factor for future rewards.
            epsilon_start (float): Initial value for epsilon (exploration rate).
            epsilon_decay (float): Factor to multiply epsilon by after each learning step/episode.
            epsilon_min (float): Minimum value epsilon can decay to.
            buffer_size (int): Capacity of the replay memory buffer.
            batch_size (int): Number of experiences to sample for each learning step.
            target_update_freq (int): Frequency (in learning steps) for hard target updates, or ignored if use_soft_update is True.
            tau (float): Interpolation factor for soft target updates.
            use_soft_update (bool): If True, use soft updates (tau). If False, use hard updates (target_update_freq).
            model_path (Optional[str]): Path to load pre-trained model weights from or save to.
            device (Optional[torch.device]): PyTorch device (CPU or CUDA). Auto-detects if None.
        """
        logger.info("Initializing DQNAgent...")

        self.board_size = board_size
        self.input_size = board_size * board_size
        self.output_size = board_size * board_size # One Q-value per cell (action)
        self.action_space = list(range(self.output_size)) # Actions represented by index 0-99

        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.tau = tau
        self.use_soft_update = use_soft_update
        self.target_update_freq = target_update_freq if not use_soft_update else 1 # Soft updates happen every learn step
        self._learn_step_counter = 0 # Counter for hard target updates

        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        logger.info(f"Using device: {self.device}")

        # Networks: Policy and Target
        self.policy_net = DQN(self.input_size, self.output_size).to(self.device)
        self.target_net = DQN(self.input_size, self.output_size).to(self.device)
        # Initialize target network with policy network's weights
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Target network is only for inference, not training
        logger.info("Policy and Target networks created.")
        logger.debug(f"Policy Net: {self.policy_net}")

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        logger.info(f"Optimizer Adam initialized with learning rate {learning_rate}.")

        # Replay Memory
        self.memory = ReplayBuffer(buffer_size)
        logger.info(f"Replay buffer initialized with size {buffer_size}.")

        # Model loading/saving path
        self.model_path = model_path
        if self.model_path:
            self.load_model(self.model_path) # Attempt to load if path exists


    def _state_to_tensor(self, state: np.ndarray) -> torch.Tensor:
        """
        Converts a NumPy game state array to a PyTorch tensor, normalizes,
        adds a batch dimension, and moves to the agent's device.

        Args:
            state (np.ndarray): The board state (e.g., opponent view), shape (H, W).

        Returns:
            torch.Tensor: The processed state tensor, shape (1, H*W), on the agent's device.
        """
        # Flatten the board state
        flat_state = state.flatten().astype(np.float32)

        # Normalize state values (optional but often helpful)
        # Example: Map values like 0, 2, 3 to a range, e.g., [0, 1] or [-1, 1]
        # Let's try a simple mapping: EMPTY=0, MISS=0.5, HIT=1.0
        normalized_state = np.zeros_like(flat_state)
        normalized_state[flat_state == BOARD_EMPTY] = 0.0
        normalized_state[flat_state == BOARD_MISS] = 0.5
        normalized_state[flat_state == BOARD_HIT] = 1.0
        # Any other state values? Assume they are 0 for now. Handle if necessary.

        # Convert to tensor, add batch dimension, move to device
        tensor = torch.tensor(normalized_state, dtype=torch.float32).unsqueeze(0).to(self.device)
        return tensor

    def _action_to_index(self, action: Tuple[int, int]) -> int:
        """Converts a (row, col) action tuple to a flat index."""
        row, col = action
        return row * self.board_size + col

    def _index_to_action(self, index: int) -> Tuple[int, int]:
        """Converts a flat action index back to a (row, col) tuple."""
        row = index // self.board_size
        col = index % self.board_size
        return row, col

    def choose_action(self, state: np.ndarray, valid_actions: List[Tuple[int, int]]) -> Tuple[int, int]:
        """
        Chooses an action using epsilon-greedy policy based on the current state
        and the set of valid (untried) actions.

        Args:
            state (np.ndarray): Current game state (opponent view).
            valid_actions (List[Tuple[int, int]]): List of possible (row, col) moves.

        Returns:
            Tuple[int, int]: The chosen action (row, col).

        Raises:
            ValueError: If `valid_actions` list is empty.
        """
        if not valid_actions:
            logger.error("Choose action called with no valid actions!")
            raise ValueError("No valid actions available to choose from.")

        # Exploration vs. Exploitation
        if random.random() < self.epsilon:
            # --- Exploration ---
            action = random.choice(valid_actions)
            logger.debug(f"Exploring: Chose random valid action {action} from {len(valid_actions)} options.")
        else:
            # --- Exploitation ---
            state_tensor = self._state_to_tensor(state)
            self.policy_net.eval() # Set network to evaluation mode for inference
            with torch.no_grad(): # Disable gradient calculations
                # Get Q-values for all possible actions (0-99) from the policy network
                all_q_values: torch.Tensor = self.policy_net(state_tensor).squeeze(0) # Shape (100,)

            self.policy_net.train() # Set network back to training mode

            # Map valid (row, col) actions to their corresponding indices (0-99)
            valid_action_indices = [self._action_to_index(act) for act in valid_actions]

            # Filter the Q-values to consider only the valid actions
            valid_q_values = all_q_values[valid_action_indices] # Tensor containing Q-values only for valid actions

            # Find the index within the *valid* Q-values that has the maximum value
            best_valid_q_index = torch.argmax(valid_q_values).item()

            # Get the corresponding action index in the full action space (0-99)
            best_action_index = valid_action_indices[best_valid_q_index]

            # Convert the best action index back to (row, col) format
            action = self._index_to_action(best_action_index)

            best_q = valid_q_values[best_valid_q_index].item()
            logger.debug(f"Exploiting: Chose action {action} (index {best_action_index}) with Q={best_q:.4f} among {len(valid_actions)} valid actions.")

        return action

    def store_transition(self, state: np.ndarray, action: Tuple[int, int], reward: float, next_state: np.ndarray, done: bool):
        """
        Stores an experience tuple in the replay buffer.
        Converts states to tensors and action to index before storing.

        Args:
            state (np.ndarray): State before action.
            action (Tuple[int, int]): Action taken (row, col).
            reward (float): Reward received.
            next_state (np.ndarray): State after action.
            done (bool): Whether the episode ended.
        """
        state_tensor = self._state_to_tensor(state)
        next_state_tensor = self._state_to_tensor(next_state)
        action_index = torch.tensor([[self._action_to_index(action)]], device=self.device, dtype=torch.long)
        reward_tensor = torch.tensor([reward], device=self.device, dtype=torch.float32)
        done_tensor = torch.tensor([done], device=self.device, dtype=torch.bool) # Use bool or float

        # Store tensors in the buffer
        self.memory.push(state_tensor, action_index, reward_tensor, next_state_tensor, done_tensor)
        # logger.debug("Stored transition in replay buffer.")


    def _update_target_network(self):
        """Updates the target network weights."""
        if self.use_soft_update:
            # Soft update: target_weights = tau * policy_weights + (1 - tau) * target_weights
            target_net_state_dict = self.target_net.state_dict()
            policy_net_state_dict = self.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)
            self.target_net.load_state_dict(target_net_state_dict)
            # logger.debug("Performed soft update of target network.")
        else:
            # Hard update: Copy policy network weights to target network
            self.target_net.load_state_dict(self.policy_net.state_dict())
            logger.debug("Performed hard update of target network.")


    def learn(self):
        """
        Samples a batch from the replay buffer and performs a Q-learning update.
        Also handles target network updates.
        """
        if len(self.memory) < self.batch_size:
            # logger.debug("Skipping learn step: not enough samples in buffer.")
            return # Not enough samples in memory

        # Sample a batch of transitions
        transitions = self.memory.sample(self.batch_size)
        if transitions is None: # Should not happen if len check passed, but defensive
            return

        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043)
        # It converts batch-array of Transitions to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Concatenate batch elements into tensors
        # Note: states and next_states are already (1, input_size), so cat(dim=0) creates (batch_size, input_size)
        state_batch = torch.cat(batch.state) # Shape: (batch_size, input_size)
        action_batch = torch.cat(batch.action_index) # Shape: (batch_size, 1)
        reward_batch = torch.cat(batch.reward) # Shape: (batch_size,)
        next_state_batch = torch.cat(batch.next_state) # Shape: (batch_size, input_size)
        done_batch = torch.cat(batch.done) # Shape: (batch_size,)


        # --- Calculate Q(s_t, a_t) ---
        # Get Q-values from the policy network for the actions that were actually taken
        # `policy_net(state_batch)` gives Q-values for all actions: (batch_size, num_actions)
        # `gather(1, action_batch)` selects the Q-value corresponding to the action taken in each state
        current_q_values = self.policy_net(state_batch).gather(1, action_batch) # Shape: (batch_size, 1)

        # --- Calculate V(s_{t+1}) for all next states ---
        # Use the target network to estimate the value of the next state.
        # We don't need gradients for this part (`torch.no_grad()`).
        with torch.no_grad():
            # `target_net(next_state_batch)` gives Q-values for all actions in the next state: (batch_size, num_actions)
            # `.max(1)` returns a tuple (max_values, max_indices) along dimension 1 (actions)
            # `.values` (or [0]) takes only the maximum Q-value for each next state
            max_next_q_values = self.target_net(next_state_batch).max(1).values # Shape: (batch_size,)

            # --- Calculate the target Q-value ---
            # If the state was terminal (done=True), the target value is just the reward.
            # Otherwise, target = reward + gamma * max_next_q
            # (1 - done_batch.float()) ensures gamma * max_next_q is zero for terminal states
            target_q_values = reward_batch + (self.gamma * max_next_q_values * (~done_batch))
            target_q_values = target_q_values.unsqueeze(1) # Reshape to (batch_size, 1) to match current_q_values

        # --- Compute the loss ---
        # Using Mean Squared Error (MSE) or Smooth L1 loss between current Q and target Q
        # loss = F.mse_loss(current_q_values, target_q_values)
        loss = F.smooth_l1_loss(current_q_values, target_q_values) # Often more robust than MSE

        # --- Optimize the model ---
        self.optimizer.zero_grad() # Clear previous gradients
        loss.backward() # Compute gradients
        # Optional: Gradient clipping to prevent exploding gradients
        # torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step() # Update network weights

        logger.debug(f"Learn step completed. Loss: {loss.item():.4f}")

        # --- Update the target network ---
        self._learn_step_counter += 1
        if self.use_soft_update:
            # Soft update happens regardless of frequency counter
             self._update_target_network()
        elif self._learn_step_counter % self.target_update_freq == 0:
             # Hard update happens based on frequency
             self._update_target_network()


    def decay_epsilon(self):
        """Decays the exploration rate (epsilon) over time."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        # logger.info(f"Decayed epsilon to {self.epsilon:.4f}") # Log maybe less frequently

    def save_model(self, file_path: Optional[str] = None):
        """
        Saves the policy network's state dictionary to a file.

        Args:
            file_path (Optional[str]): Path to save the model. Uses agent's path if None.
        """
        path = file_path or self.model_path
        if not path:
            logger.warning("Cannot save model: No file path specified.")
            return

        try:
            directory = os.path.dirname(path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            torch.save(self.policy_net.state_dict(), path)
            logger.info(f"Policy network state saved to {path}")
        except (OSError, Exception) as e:
            logger.error(f"Error saving model to {path}: {e}")

    def load_model(self, file_path: str):
        """
        Loads the policy network's state dictionary from a file and copies it
        to the target network.

        Args:
            file_path (str): Path to load the model from.
        """
        if not os.path.exists(file_path):
            logger.warning(f"Model file not found at {file_path}. Starting with initial weights.")
            return

        try:
            # Load state dict, ensuring it's mapped to the correct device
            state_dict = torch.load(file_path, map_location=self.device)
            self.policy_net.load_state_dict(state_dict)
            self.target_net.load_state_dict(state_dict) # Keep target net consistent
            self.policy_net.train() # Ensure policy net is in train mode after loading
            self.target_net.eval()  # Ensure target net is in eval mode
            logger.info(f"Successfully loaded model state from {path}")
        except (RuntimeError, FileNotFoundError, Exception) as e:
            logger.error(f"Error loading model from {file_path}: {e}. Using initial weights.")


# Example Usage Block (for standalone testing)
if __name__ == "__main__":
    # Configure logging for this test run
    logging.basicConfig(level=logging.DEBUG, # Show debug messages
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        force=True)
    logger.info("Starting DQNAgent example usage (standalone test)...")

    # --- Agent Initialization ---
    test_board_size = 4 # Smaller board for quicker testing
    temp_model_file = 'test_dqn_model.pth'

    agent = DQNAgent(
        board_size=test_board_size,
        learning_rate=0.001,
        gamma=0.9,
        epsilon_start=0.9,
        epsilon_decay=0.99,
        epsilon_min=0.05,
        buffer_size=1000, # Smaller buffer for testing
        batch_size=32,    # Smaller batch size
        target_update_freq=5,
        use_soft_update=False, # Test hard update
        model_path=temp_model_file
    )

    # Ensure any previous test file is removed
    if os.path.exists(temp_model_file):
        os.remove(temp_model_file)
        logger.info(f"Removed existing {temp_model_file} for fresh test.")

    # --- Simulate First Step ---
    # Initial state: 4x4 board, all empty (value 0)
    current_state_np = np.full((test_board_size, test_board_size), BOARD_EMPTY, dtype=int)
    # Valid actions: All squares are initially valid
    valid_actions_list = [(r, c) for r in range(test_board_size) for c in range(test_board_size)]

    print("\n--- Choosing First Action ---")
    try:
        chosen_action = agent.choose_action(current_state_np, valid_actions_list)
        print(f"State:\n{current_state_np}")
        print(f"Valid Actions ({len(valid_actions_list)}): {valid_actions_list}")
        print(f"Chosen Action: {chosen_action}")
        print(f"Current Epsilon: {agent.epsilon:.4f}")
    except ValueError as e:
        print(f"Error choosing action: {e}")
        exit()

    # --- Simulate Storing Transition ---
    reward_value = -0.1 # Assume a miss
    next_state_np = current_state_np.copy()
    next_state_np[chosen_action] = BOARD_MISS # Mark the miss
    game_done = False

    print("\n--- Storing First Transition ---")
    agent.store_transition(current_state_np, chosen_action, reward_value, next_state_np, game_done)
    print(f"Buffer size: {len(agent.memory)}")

    # --- Simulate filling buffer and learning ---
    print("\n--- Simulating Buffer Filling & Learning ---")
    # Fill buffer with enough dummy transitions for one batch
    for i in range(agent.batch_size - 1): # Already stored one transition
        dummy_state = np.random.randint(0, 4, size=(test_board_size, test_board_size)) # Random states
        dummy_next = np.random.randint(0, 4, size=(test_board_size, test_board_size))
        dummy_action = random.choice(valid_actions_list)
        dummy_reward = random.uniform(-1, 1)
        dummy_done = random.choice([True, False])
        agent.store_transition(dummy_state, dummy_action, dummy_reward, dummy_next, dummy_done)

    print(f"Buffer size after filling: {len(agent.memory)}")
    print("Attempting first learn step...")
    agent.learn() # Perform a learning step

    # Check if target network update occurs after enough steps
    print("\n--- Simulating more steps for target update ---")
    for _ in range(agent.target_update_freq * 2): # Ensure update is triggered
        # Add one more transition to keep buffer full enough if needed
        dummy_state = np.random.randint(0, 4, size=(test_board_size, test_board_size))
        dummy_next = np.random.randint(0, 4, size=(test_board_size, test_board_size))
        dummy_action = random.choice(valid_actions_list)
        agent.store_transition(dummy_state, dummy_action, 0, dummy_next, False)
        agent.learn()


    # --- Decay Epsilon ---
    print("\n--- Decaying Epsilon ---")
    agent.decay_epsilon()
    print(f"Epsilon after decay: {agent.epsilon:.4f}")

    # --- Test Save and Load Functionality ---
    print("\n--- Saving Model ---")
    agent.save_model() # Saves to 'test_dqn_model.pth'

    print("\n--- Loading Model into new agent ---")
    agent_loaded = DQNAgent(board_size=test_board_size, model_path=temp_model_file)

    # Verify loaded weights are different from initial weights (check bias of first layer)
    initial_bias = DQN(agent.input_size, agent.output_size).fc1.bias.detach().cpu().numpy()
    loaded_bias = agent_loaded.policy_net.fc1.bias.detach().cpu().numpy()
    print(f"Bias comparison (sample): Initial={initial_bias[0]}, Loaded={loaded_bias[0]}")
    assert not np.allclose(initial_bias, loaded_bias), "Loaded model seems identical to a newly initialized one."

    # --- Cleanup ---
    if os.path.exists(temp_model_file):
        os.remove(temp_model_file)
        print(f"\nCleaned up temporary file: {temp_model_file}")

    logger.info("Finished DQNAgent example usage.")