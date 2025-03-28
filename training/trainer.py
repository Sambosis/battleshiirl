# Generated with write_code_to_file
# /home/myuser/apps/battleshiprl/training/trainer.py

import time
import logging
import random
import os
import pygame # Need for event handling during visualization

# Import project modules
try:
    from battleship.game import BattleshipGame, BOARD_EMPTY, BOARD_HIT, BOARD_MISS
    from models.q_learning_agent import QLearningAgent
    from models.dqn_agent import DQNAgent
    from visualization.renderer import GameRenderer
    from utils.metrics import TrainingMetrics
except ImportError as e:
    logging.error(f"Error importing project modules: {e}. Check PYTHONPATH or project structure.")
    # Depending on setup, may need to add parent directory to sys.path if running directly
    import sys
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)
    try:
        from battleship.game import BattleshipGame, BOARD_EMPTY, BOARD_HIT, BOARD_MISS
        from models.q_learning_agent import QLearningAgent
        from models.dqn_agent import DQNAgent
        from visualization.renderer import GameRenderer
        from utils.metrics import TrainingMetrics
    except ImportError as inner_e:
        logging.error(f"Still failed to import after path adjustment: {inner_e}")
        sys.exit("Critical import error. Cannot run trainer.")


# Set up logging
# Configuration should ideally happen in main.py, but ensure logger exists
logger = logging.getLogger(__name__)
# Basic config if run standalone or not configured elsewhere:
if not logger.hasHandlers():
     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class Trainer:
    """
    Manages the training process for two RL agents playing Battleship against each other.
    Handles episodes, agent updates, metrics tracking, and visualization.
    """

    # --- Reward Configuration ---
    REWARD_HIT = 2.0
    REWARD_MISS = -0.5 # Penalize misses to encourage efficiency
    REWARD_SINK = 5.0  # Bonus for sinking a ship
    REWARD_WIN = 20.0 # Large bonus for winning the game
    REWARD_LOSE = -20.0 # Negative bonus for losing
    REWARD_INVALID = -10.0 # Penalty for invalid moves (should be rare with agent logic)
    REWARD_ALREADY_FIRED = -2.0 # Penalize shooting at known spots

    def __init__(self,
                 num_episodes: int,
                 visualize_every_n: int = 5,
                 save_every_n: int = 100,
                 agent0_config: dict = {}, # Configuration for player 0
                 agent1_config: dict = {}, # Configuration for player 1
                 q_agent_save_path: str = 'q_learning_agent.pkl',
                 dqn_agent_save_path: str = 'dqn_agent.pth',
                 metrics_save_path: str = 'training_metrics.json'
                 ):
        """
        Initializes the Trainer.

        Args:
            num_episodes (int): Total number of games (episodes) to train for.
            visualize_every_n (int): Frequency of visualizing games (e.g., visualize every 5th game). Set to 0 to disable.
            save_every_n (int): Frequency of saving agent models (e.g., save every 100th game). Set to 0 to disable.
            agent0_config (dict): Configuration for agent 0 (e.g., {'type': 'q_learning', 'lr': 0.1, ...}).
            agent1_config (dict): Configuration for agent 1 (e.g., {'type': 'dqn', 'buffer_size': 10000, ...}).
            q_agent_save_path (str): File path to save/load the Q-learning agent's Q-table.
            dqn_agent_save_path (str): File path to save/load the DQN agent's model weights.
            metrics_save_path (str): File path to save tracked training metrics.
        """
        logger.info("Initializing Trainer...")

        self.num_episodes = num_episodes
        self.visualize_every_n = visualize_every_n
        self.save_every_n = save_every_n
        self.q_agent_save_path = q_agent_save_path
        self.dqn_agent_save_path = dqn_agent_save_path
        self.metrics_save_path = metrics_save_path

        # Initialize Game Environment
        self.game = BattleshipGame()
        self.board_size = self.game.board_size
        logger.info(f"Battleship game initialized with board size {self.board_size}x{self.board_size}.")

        # Initialize Agents based on config
        self.agents = {
            0: self._create_agent(0, agent0_config),
            1: self._create_agent(1, agent1_config)
        }
        logger.info(f"Agent 0 Type: {type(self.agents[0]).__name__}")
        logger.info(f"Agent 1 Type: {type(self.agents[1]).__name__}")

        # Initialize Renderer (only if needed)
        self.renderer = None
        if self.visualize_every_n > 0:
            try:
                self.renderer = GameRenderer()
            except Exception as e:
                 logger.error(f"Failed to initialize GameRenderer: {e}. Visualization disabled.")
                 self.visualize_every_n = 0 # Disable visualization if init fails


        # Initialize Metrics Tracking
        # Pass agent types for labeling in metrics
        agent_types = {p: type(a).__name__ for p, a in self.agents.items()}
        self.metrics = TrainingMetrics(agent_types=agent_types, save_path=self.metrics_save_path)
        logger.info("TrainingMetrics initialized.")

        # Ensure save directories exist
        self._ensure_save_dir(self.q_agent_save_path)
        self._ensure_save_dir(self.dqn_agent_save_path)
        self._ensure_save_dir(self.metrics_save_path)

    def _ensure_save_dir(self, file_path: str):
        """Creates the directory for a file path if it doesn't exist."""
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
                logger.info(f"Created directory for saving: {directory}")
            except OSError as e:
                logger.error(f"Could not create directory {directory}: {e}")

    def _create_agent(self, player_id: int, config: dict):
        """Factory method to create an agent based on the configuration."""
        agent_type = config.get('type', 'q_learning').lower() # Default to Q-learning
        logger.info(f"Creating Agent {player_id} of type: {agent_type}")

        if agent_type == 'q_learning':
            # Extract Q-learning specific params from config or use defaults
            params = {
                'board_size': self.board_size,
                'learning_rate': config.get('learning_rate', 0.1),
                'discount_factor': config.get('discount_factor', 0.95),
                'epsilon': config.get('epsilon_start', 1.0),
                'epsilon_decay': config.get('epsilon_decay', 0.9995),
                'min_epsilon': config.get('min_epsilon', 0.01),
                'q_table_path': self.q_agent_save_path # Use standardized path
            }
            logger.debug(f"QLearningAgent params for Player {player_id}: {params}")
            return QLearningAgent(**params)

        elif agent_type == 'dqn':
            # Extract DQN specific params from config or use defaults
            params = {
                'board_size': self.board_size,
                'learning_rate': config.get('learning_rate', 0.0005),
                'gamma': config.get('gamma', 0.99),
                'epsilon_start': config.get('epsilon_start', 1.0),
                'epsilon_decay': config.get('epsilon_decay', 0.9995),
                'epsilon_min': config.get('min_epsilon', 0.05),
                'buffer_size': config.get('buffer_size', 10000),
                'batch_size': config.get('batch_size', 64),
                'target_update_freq': config.get('target_update_freq', 10),
                'tau': config.get('tau', 0.005),
                'use_soft_update': config.get('use_soft_update', True),
                'model_path': self.dqn_agent_save_path # Use standardized path
            }
            logger.debug(f"DQNAgent params for Player {player_id}: {params}")
            return DQNAgent(**params)
        else:
            raise ValueError(f"Unknown agent type '{agent_type}' specified in configuration.")

    def _get_reward(self, result: dict, agent_id: int) -> float:
        """
        Calculates the reward based on the outcome of a 'fire' action.

        Args:
            result (dict): The dictionary returned by `game.fire()`.
            agent_id (int): The ID of the agent who performed the action.

        Returns:
            float: The calculated reward value.
        """
        outcome = result.get('result', 'invalid')
        sunk_ship = result.get('sunk')
        game_over = result.get('game_over', False)
        winner = result.get('winner', None) # Winner is only set when game_over is True

        reward = 0.0

        if outcome == 'hit':
            reward += self.REWARD_HIT
            if sunk_ship:
                reward += self.REWARD_SINK
                logger.debug(f"Reward: Sink bonus {self.REWARD_SINK} added for sinking {sunk_ship}")
        elif outcome == 'miss':
            reward += self.REWARD_MISS
        elif outcome == 'already_fired':
            reward += self.REWARD_ALREADY_FIRED
            logger.warning(f"Agent {agent_id} fired at already targeted location {result['coordinate']}. Penalty applied.")
        elif outcome == 'invalid':
            reward += self.REWARD_INVALID
            logger.error(f"Agent {agent_id} made an invalid move to {result['coordinate']}. Penalty applied.") # Should be rare

        # Check for win/loss condition, which occurs *after* the move resolves
        if game_over:
            if winner == agent_id:
                reward += self.REWARD_WIN
                logger.info(f"Reward: Win bonus {self.REWARD_WIN} added for Agent {agent_id}")
            else:
                # The agent who just moved was NOT the winner, meaning they lost
                # (or potentially a draw, but Battleship doesn't typically have draws)
                reward += self.REWARD_LOSE
                # This reward is applied to the final action of the *losing* agent
                # which might seem odd, but it propagates the negative outcome.
                # However, the win reward for the winner on their final move is arguably more impactful.
                # Let's reconsider: Assign WIN only to the winner's final move.
                # Don't assign LOSE here, the absence of WIN and potential negative action rewards serve as penalty.
                # Let's stick to the original: assign WIN if winner == agent_id.
                # If the game ends due to *this* agent's move and they are NOT the winner,
                # it means they just got hit and lost. This reward calculation is happening
                # for the agent *whose turn it was*.
                # RETHINK: The reward is for the agent who *took the action*.
                # If agent A takes action, results in game over, and agent A is winner: reward += WIN
                # If agent A takes action, results in game over, and agent B is winner: This scenario shouldn't happen directly.
                # Game over occurs because agent B has no ships left *after* A's hit.
                # So, the win reward is naturally associated with the winning move.
                pass # No explicit lose reward added here, loss is implicit in metrics & not getting win reward

        logger.debug(f"Calculated reward for Agent {agent_id}, Action: {result['coordinate']}, Outcome: {outcome}, Sunk: {sunk_ship}, GameOver: {game_over}, Winner: {winner} -> Reward: {reward:.2f}")
        return reward


    def _run_episode(self, episode_num: int) -> dict:
        """
        Runs a single episode (game) of Battleship between the two agents.

        Args:
            episode_num (int): The current episode number.

        Returns:
            dict: A dictionary containing episode results (e.g., 'winner', 'turns').
        """
        logger.info(f"--- Starting Episode {episode_num} ---")
        self.game.reset()
        start_time = time.time()
        visualize = (self.visualize_every_n > 0 and episode_num % self.visualize_every_n == 0) and self.renderer is not None

        game_states = [] # Store (state, action, reward) for debugging/analysis if needed
        total_steps = 0

        last_state = {0: None, 1: None}
        last_action = {0: None, 1: None}

        # Initial visualization
        if visualize:
            if not self.renderer.render(self.game):
                logger.warning("Visualization window closed by user during episode start.")
                return {'winner': None, 'turns': 0, 'duration': 0, 'aborted': True} # Signal abort


        while not self.game.game_over:
            current_player = self.game.current_player
            agent = self.agents[current_player]

            # 1. Get current state and valid actions
            current_state_np = self.game.get_state(current_player)
            valid_actions = self.game.get_valid_actions(current_player)

            if not valid_actions:
                # This should ideally not happen unless the game ends unexpectedly or board full?
                logger.error(f"Episode {episode_num}: No valid actions for Player {current_player}, but game not over? Ending potentially incomplete episode.")
                break

            # Store the state *before* the action is chosen (for the previous player's update)
            # This seems overly complex. Let's stick to the standard RL loop:
            # Get state S, choose action A, execute A, get reward R and next state S'. Store (S, A, R, S').

            # 2. Agent chooses action
            chosen_action = agent.choose_action(current_state_np, valid_actions)
            logger.debug(f"Episode {episode_num}, Turn {self.game.turns_taken}: Player {current_player} ({type(agent).__name__}) chose action {chosen_action}")

            # --- Store state S and action A before executing ---
            state_before_action = current_state_np

            # 3. Execute action in game
            result = self.game.fire(chosen_action) # This switches player internally if needed

            # 4. Calculate reward for the action taken
            reward = self._get_reward(result, current_player)

            # 5. Get the next state (for the player who just acted)
            # The 'next state' is the board view *after* their shot resolved.
            next_state_np = self.game.get_state(current_player)
            done = result.get('game_over', False)

            # 6. Store transition and update the agent who just acted
            if isinstance(agent, QLearningAgent):
                # Q-learning updates directly using the tuple
                agent.update(state_before_action, chosen_action, reward, next_state_np, done)
            elif isinstance(agent, DQNAgent):
                # DQN stores transition in buffer, then potentially learns from a batch
                agent.store_transition(state_before_action, chosen_action, reward, next_state_np, done)
                # Trigger learning step (e.g., every step, or check buffer size)
                agent.learn() # Assume learn handles buffer size check internally

            # Store for potential analysis (optional)
            game_states.append({'turn': self.game.turns_taken, 'player': current_player, 'action': chosen_action, 'result': result, 'reward': reward})
            total_steps = self.game.turns_taken # Use game's turn count

            # 7. Visualize if enabled
            if visualize:
                 if not self.renderer.render(self.game):
                     logger.warning("Visualization window closed by user during episode.")
                     return {'winner': None, 'turns': total_steps, 'duration': time.time() - start_time, 'aborted': True}
                 pygame.time.wait(150) # Slow down visualization slightly (in ms)

            # Check for game termination (redundant with loop condition but safe)
            if done:
                 logger.info(f"Episode {episode_num} finished. Winner: {self.game.winner}")
                 break

        # --- Episode End ---
        duration = time.time() - start_time
        winner = self.game.winner
        turns = self.game.turns_taken

        # Final visualization frame if needed
        if visualize:
            if not self.renderer.render(self.game):
                 logger.warning("Visualization window closed post-game.")
                 return {'winner': winner, 'turns': turns, 'duration': duration, 'aborted': True}
            pygame.time.wait(1000) # Pause on final screen

        # Decay epsilon for both agents after the episode
        self.agents[0].decay_epsilon()
        self.agents[1].decay_epsilon()
        logger.info(f"Episode {episode_num} ended. Winner: {winner}, Turns: {turns}, Duration: {duration:.2f}s")
        logger.info(f"Epsilons: P0={self.agents[0].epsilon:.4f}, P1={self.agents[1].epsilon:.4f}")

        return {'winner': winner, 'turns': turns, 'duration': duration, 'aborted': False}


    def train(self):
        """
        Starts and manages the main training loop over the specified number of episodes.
        """
        logger.info(f"=== Starting Training for {self.num_episodes} Episodes ===")
        start_total_time = time.time()

        aborted = False
        for i in range(1, self.num_episodes + 1):
            episode_result = self._run_episode(i)

            if episode_result['aborted']:
                logger.warning(f"Training aborted during episode {i} due to visualization closure.")
                aborted = True
                break

            # Record metrics for the completed episode
            self.metrics.record_episode(
                episode_num=i,
                winner=episode_result['winner'],
                turns=episode_result['turns'],
                duration=episode_result['duration'],
                epsilon0=self.agents[0].epsilon,
                epsilon1=self.agents[1].epsilon
            )

            # Print progress periodically
            if i % 10 == 0 or i == self.num_episodes: # Print every 10 episodes and at the end
                self.metrics.print_progress(i, self.num_episodes, start_total_time)

            # Save models periodically
            if self.save_every_n > 0 and i % self.save_every_n == 0:
                logger.info(f"--- Saving agent models at episode {i} ---")
                if isinstance(self.agents[0], QLearningAgent) or isinstance(self.agents[1], QLearningAgent):
                    q_agent = self.agents[0] if isinstance(self.agents[0], QLearningAgent) else self.agents[1]
                    if q_agent: q_agent.save_q_table(self.q_agent_save_path)
                if isinstance(self.agents[0], DQNAgent) or isinstance(self.agents[1], DQNAgent):
                    dqn_agent = self.agents[0] if isinstance(self.agents[0], DQNAgent) else self.agents[1]
                    if dqn_agent: dqn_agent.save_model(self.dqn_agent_save_path)
                # Save metrics as well
                self.metrics.save_metrics()


        # --- Training Finished ---
        end_total_time = time.time()
        total_training_time = end_total_time - start_total_time
        logger.info(f"=== Training Finished ({'Aborted' if aborted else 'Completed'}) ===")
        logger.info(f"Total episodes run: {i if aborted else self.num_episodes}")
        logger.info(f"Total training time: {total_training_time:.2f} seconds")

        # Save final models and metrics
        logger.info("--- Saving final agent models and metrics ---")
        if isinstance(self.agents[0], QLearningAgent) or isinstance(self.agents[1], QLearningAgent):
            q_agent = self.agents[0] if isinstance(self.agents[0], QLearningAgent) else self.agents[1]
            if q_agent: q_agent.save_q_table(self.q_agent_save_path)
        if isinstance(self.agents[0], DQNAgent) or isinstance(self.agents[1], DQNAgent):
             dqn_agent = self.agents[0] if isinstance(self.agents[0], DQNAgent) else self.agents[1]
             if dqn_agent: dqn_agent.save_model(self.dqn_agent_save_path)
        self.metrics.save_metrics()

        # Plot final metrics if possible
        try:
            self.metrics.plot_metrics()
            logger.info("Metrics plots generated (if matplotlib available).")
        except Exception as e:
            logger.warning(f"Could not generate metrics plots: {e}")

        # Close renderer if it was used
        if self.renderer:
            self.renderer.close()


# Example Usage (typically called from main.py)
if __name__ == "__main__":
    # Configure logging for standalone run
    logging.basicConfig(level=logging.DEBUG, # More verbose for testing
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        force=True)

    logger.info("Starting Trainer standalone test...")

    # Define configurations for the agents
    # Example: Q-learning vs DQN
    agent0_cfg = {
        'type': 'q_learning',
        'learning_rate': 0.1,
        'discount_factor': 0.9,
        'epsilon_start': 1.0,
        'epsilon_decay': 0.999, # Faster decay for short test
        'min_epsilon': 0.05
    }
    agent1_cfg = {
        'type': 'dqn',
        'learning_rate': 0.001,
        'gamma': 0.95,
        'epsilon_start': 1.0,
        'epsilon_decay': 0.998, # Faster decay for short test
        'epsilon_min': 0.05,
        'buffer_size': 5000, # Smaller buffer for quick test
        'batch_size': 32,
        'target_update_freq': 5
    }

    # Paths for saving
    test_q_save = 'temp_test_q_agent.pkl'
    test_dqn_save = 'temp_test_dqn_agent.pth'
    test_metrics_save = 'temp_test_metrics.json'

    # Clean up previous test files
    if os.path.exists(test_q_save): os.remove(test_q_save)
    if os.path.exists(test_dqn_save): os.remove(test_dqn_save)
    if os.path.exists(test_metrics_save): os.remove(test_metrics_save)


    trainer = Trainer(
        num_episodes=25,       # Run a small number of episodes for testing
        visualize_every_n=5,   # Visualize every 5th game
        save_every_n=10,       # Save models every 10 games
        agent0_config=agent0_cfg,
        agent1_config=agent1_cfg,
        q_agent_save_path=test_q_save,
        dqn_agent_save_path=test_dqn_save,
        metrics_save_path=test_metrics_save
    )

    try:
        trainer.train()
    except Exception as e:
         logger.error(f"An error occurred during training: {e}", exc_info=True)
    finally:
         # Clean up test files after run (optional)
         # if os.path.exists(test_q_save): os.remove(test_q_save)
         # if os.path.exists(test_dqn_save): os.remove(test_dqn_save)
         # if os.path.exists(test_metrics_save): os.remove(test_metrics_save)
         pass # Keep files for inspection after test

    logger.info("Trainer standalone test finished.")

# Potential improvements:
# - More sophisticated reward shaping (e.g., potential-based rewards).
# - Hyperparameter tuning mechanism.
# - Allowing different agent types for P0 and P1 dynamically. (Current setup requires specifying paths for both, even if only one type is used). -> Fixed by checking instance type before saving.
# - More robust handling of visualization closure (e.g., pausing training).
# - Option to configure device (CPU/GPU) for DQN agent via trainer config.
# - Checkpointing: Saving trainer state (episode number, agent states, optimizer states) to allow resuming training.
# - Parallel training (more complex).
# - More detailed logging of agent-specific parameters (e.g., DQN buffer fill rate).