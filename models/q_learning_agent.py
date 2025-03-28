import numpy as np
import random
import pickle
import os
import logging
from typing import Dict, Tuple, List, Optional, Any

# Import game constants
try:
    from battleship.game import BOARD_SIZE, BOARD_EMPTY, BOARD_HIT, BOARD_MISS
except ImportError:
    # If running outside the project structure or for testing
    logging.warning("Could not import from battleship.game, using default values")
    BOARD_SIZE = 10
    BOARD_EMPTY = 0
    BOARD_HIT = 2
    BOARD_MISS = 3

# Type aliases for improved readability
StateKey = Tuple[int, ...]  # Flattened board state as tuple
Action = Tuple[int, int]    # (row, col) coordinate

# Set up logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class QLearningAgent:
    """
    Implements a Q-learning agent to play Battleship.
    Uses a Q-table (dictionary) to learn State-Action values (Q-values).
    The state is represented by the agent's view of the opponent's board.
    """

    def __init__(self, board_size: int = BOARD_SIZE, learning_rate: float = 0.1, 
                 discount_factor: float = 0.95, epsilon: float = 1.0, 
                 epsilon_decay: float = 0.9995, min_epsilon: float = 0.01, 
                 q_table_path: Optional[str] = None):
        """
        Initializes the Q-learning agent.
        
        Args:
            board_size (int): The dimension of the game board (e.g., 10 for 10x10).
            learning_rate (float): Alpha, the learning rate determining how much new information overrides old.
            discount_factor (float): Gamma, the discount factor for future rewards.
            epsilon (float): Initial probability of choosing a random action (exploration).
            epsilon_decay (float): The factor by which epsilon is multiplied after each episode or step to reduce exploration over time.
            min_epsilon (float): The minimum value epsilon can decay to.
            q_table_path (Optional[str]): Path to load an existing Q-table from or save the learned table to.
        """
        self.board_size = board_size
        self.learning_rate = learning_rate  # alpha
        self.discount_factor = discount_factor  # gamma
        self.epsilon = epsilon  # exploration rate
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table_path = q_table_path
        
        # Initialize an empty Q-table as a nested dictionary: {state_key: {action: q_value}}
        self.q_table: Dict[StateKey, Dict[Action, float]] = {}
        
        # Load Q-table from file if provided
        if q_table_path and os.path.exists(q_table_path):
            self.load_q_table(q_table_path)
            logger.info(f"Loaded Q-table from {q_table_path}")
        else:
            logger.info("Starting with a new Q-table")
        
        # Track learning statistics
        self.update_count = 0
        
        logger.info(f"Initialized Q-learning agent with parameters: alpha={learning_rate}, gamma={discount_factor}, "
                   f"epsilon={epsilon}, epsilon_decay={epsilon_decay}, min_epsilon={min_epsilon}")

    def _get_state_key(self, state: np.ndarray) -> StateKey:
        """
        Converts the NumPy board state array into a hashable tuple key for the Q-table.
        
        Args:
            state (np.ndarray): The board state (typically the agent's 'opponent view' showing hits/misses).
                                Expected shape: (board_size, board_size).
        
        Returns:
            StateKey: A tuple representation of the flattened board state, suitable for dictionary keys.
        """
        # Flatten the 2D array to 1D and convert to tuple for hashability
        return tuple(state.flatten())

    def _initialize_state(self, state_key: StateKey):
        """
        Ensures a state exists in the Q-table. If not, initializes Q-values
        for all possible actions in that state to 0.0.
        
        Args:
            state_key (StateKey): The hashable state key.
        """
        if state_key not in self.q_table:
            # Create a new dictionary for this state
            self.q_table[state_key] = {}
            
            # Initialize Q-values for all potential actions to 0.0
            # Note: We don't need to initialize all possible (row, col) combinations here,
            # as we'll dynamically add them when encountered as valid actions
            logger.debug(f"Initialized new state in Q-table: {state_key[:5]}...{state_key[-5:]} (showing first/last 5 elements)")

    def choose_action(self, state: np.ndarray, valid_actions: List[Action]) -> Action:
        """
        Chooses an action based on the current state using the epsilon-greedy strategy.
        Explores by choosing a random valid action with probability epsilon,
        exploits by choosing the valid action with the highest Q-value otherwise.
        
        Args:
            state (np.ndarray): The current game state (agent's view of the opponent's board).
            valid_actions (List[Action]): A list of coordinates (actions) that are currently allowed
                                          (typically, squares not yet fired upon).
        
        Returns:
            Action: The chosen action (row, col).
        
        Raises:
            ValueError: If `valid_actions` list is empty.
        """
        if not valid_actions:
            raise ValueError("No valid actions available")
        
        # Convert state to a hashable key
        state_key = self._get_state_key(state)
        
        # Ensure state exists in Q-table
        self._initialize_state(state_key)
        
        # EXPLORE: Choose random action with probability epsilon
        if random.random() < self.epsilon:
            chosen_action = random.choice(valid_actions)
            logger.debug(f"EXPLORE: Randomly chose action {chosen_action}")
            return chosen_action
        
        # EXPLOIT: Choose action with highest Q-value among valid actions
        
        # 1. Get Q-values for all valid actions for this state
        q_values_for_valid_actions = {}
        for action in valid_actions:
            # If this action hasn't been tried before for this state, initialize its Q-value to 0.0
            if action not in self.q_table[state_key]:
                self.q_table[state_key][action] = 0.0
            
            q_values_for_valid_actions[action] = self.q_table[state_key][action]
        
        # 2. Find actions with the maximum Q-value (might be multiple)
        max_q_value = max(q_values_for_valid_actions.values(), default=0.0)
        best_actions = [action for action, q_value in q_values_for_valid_actions.items() 
                      if q_value == max_q_value]
        
        # 3. If multiple actions have the same max Q-value, randomly choose one
        chosen_action = random.choice(best_actions)
        logger.debug(f"EXPLOIT: Chose action {chosen_action} with Q-value {max_q_value}")
        
        return chosen_action

    def update(self, state: np.ndarray, action: Action, reward: float, next_state: np.ndarray, done: bool):
        """
        Updates the Q-value for the given state-action pair using the Q-learning formula:
        Q(s, a) <- Q(s, a) + alpha * [reward + gamma * max_a'(Q(s', a')) - Q(s, a)]
        
        Args:
            state (np.ndarray): The state before the action was taken.
            action (Action): The action taken.
            reward (float): The reward received after taking the action.
            next_state (np.ndarray): The resulting state after the action was taken.
            done (bool): True if the episode/game ended after this action, False otherwise.
        """
        # Convert states to hashable keys
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)
        
        # Ensure both states exist in the Q-table
        self._initialize_state(state_key)
        if not done:  # No need to initialize next_state if episode is done
            self._initialize_state(next_state_key)
        
        # If this is the first time seeing this action for this state, initialize its Q-value
        if action not in self.q_table[state_key]:
            self.q_table[state_key][action] = 0.0
        
        # Get current Q-value for the state-action pair
        current_q = self.q_table[state_key][action]
        
        # Calculate max future Q-value from next_state
        if done:
            # If the episode is done, there is no future reward
            max_future_q = 0.0
        else:
            # If next_state has no actions yet, max_future_q will be 0.0
            max_future_q = max(self.q_table[next_state_key].values(), default=0.0)
        
        # Q-learning formula: Q(s,a) = Q(s,a) + alpha * [r + gamma * max_a'(Q(s',a')) - Q(s,a)]
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_future_q - current_q)
        
        # Update Q-value in the table
        self.q_table[state_key][action] = new_q
        
        # Update statistics
        self.update_count += 1
        if self.update_count % 1000 == 0:
            logger.debug(f"Q-learning update #{self.update_count}: state {state_key[:3]}..., action {action}, "
                       f"reward {reward}, new_q {new_q:.4f}")

    def decay_epsilon(self):
        """
        Decays the exploration rate (epsilon) according to the decay factor,
        ensuring it doesn't fall below the minimum epsilon value.
        Should be called periodically (e.g., end of each episode).
        """
        # Apply decay
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        logger.debug(f"Decayed exploration rate (epsilon) to {self.epsilon:.6f}")

    def save_q_table(self, file_path: Optional[str] = None):
        """
        Saves the current Q-table dictionary to a file using pickle serialization.
        
        Args:
            file_path (Optional[str]): The path to save the file. If None, uses the
                                       `q_table_path` specified during initialization.
        """
        if file_path is None:
            file_path = self.q_table_path
        
        if file_path is None:
            logger.warning("No file path provided to save Q-table, skipping save")
            return
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(self.q_table, f)
            logger.info(f"Successfully saved Q-table to {file_path} ({len(self.q_table)} states)")
        except (IOError, pickle.PickleError) as e:
            logger.error(f"Error saving Q-table to {file_path}: {e}")
    
    def load_q_table(self, file_path: str):
        """
        Loads a Q-table dictionary from a file using pickle.
        If the file doesn't exist or is invalid, logs a warning/error
        and continues with an empty Q-table.
        
        Args:
            file_path (str): The path to load the Q-table file from.
        """
        try:
            with open(file_path, 'rb') as f:
                loaded_data = pickle.load(f)
            
            # Validate loaded data is a dictionary
            if not isinstance(loaded_data, dict):
                logger.error(f"Loaded data from {file_path} is not a dictionary")
                return
            
            self.q_table = loaded_data
            logger.info(f"Successfully loaded Q-table from {file_path} ({len(self.q_table)} states)")
            
        except FileNotFoundError:
            logger.warning(f"Q-table file {file_path} not found, starting with a new Q-table")
        except (IOError, pickle.PickleError) as e:
            logger.error(f"Error loading Q-table from {file_path}: {e}")


# Example usage for testing/debugging
if __name__ == "__main__":
    # Set up logging for standalone testing
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', force=True)
    
    # Create a Q-learning agent
    agent = QLearningAgent(
        board_size=5,  # Smaller board for testing
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=0.5,
        epsilon_decay=0.99,
        min_epsilon=0.01
    )
    
    # Simulate some states and actions
    logger.info("Testing Q-learning agent with simulated states/actions")
    
    # Create a simple test state
    test_state = np.full((5, 5), BOARD_EMPTY)
    test_state[1, 2] = BOARD_HIT
    test_state[3, 3] = BOARD_MISS
    
    # List of valid actions
    valid_actions = [(i, j) for i in range(5) for j in range(5) 
                    if (i, j) not in [(1, 2), (3, 3)]]
    
    # Test choosing an action
    for _ in range(3):
        action = agent.choose_action(test_state, valid_actions)
        logger.info(f"Chose action: {action}")
    
    # Test updating the Q-table
    next_state = test_state.copy()
    next_state[2, 2] = BOARD_HIT  # Assume we hit something
    
    agent.update(test_state, (2, 2), 1.0, next_state, False)
    
    # Test epsilon decay
    logger.info(f"Initial epsilon: {agent.epsilon}")
    for _ in range(10):
        agent.decay_epsilon()
    logger.info(f"Epsilon after 10 decays: {agent.epsilon}")
    
    # Test saving and loading
    temp_file = "test_q_table.pkl"
    try:
        agent.save_q_table(temp_file)
        
        # Create new agent and load Q-table
        new_agent = QLearningAgent(board_size=5, q_table_path=temp_file)
        
        # Check if both agents have the same Q-values
        state_key = agent._get_state_key(test_state)
        action = (2, 2)
        
        if state_key in new_agent.q_table and action in new_agent.q_table[state_key]:
            logger.info(f"Verified Q-value for test state and action: "
                      f"{new_agent.q_table[state_key][action]}")
    
    finally:
        # Clean up test file
        if os.path.exists(temp_file):
            os.remove(temp_file)