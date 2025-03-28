# Generated with write_code_to_file
# /home/myuser/apps/battleshiprl/main.py

import argparse
import logging
import os
import sys
import time
from typing import Dict, Any, Optional, Tuple

# Set project root for reliable imports
PROJECT_ROOT = "/home/myuser/apps/battleshiprl"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
    print(f"Info: Added {PROJECT_ROOT} to sys.path")

# Try importing project modules
try:
    from training.trainer import Trainer
    # These might not be strictly needed in main.py if Trainer handles agent creation,
    # but good for type checking or direct instantiation if needed later.
    from models.q_learning_agent import QLearningAgent
    from models.dqn_agent import DQNAgent
    # Import game and metrics for potential use or context
    from battleship.game import BattleshipGame
    from utils.metrics import TrainingMetrics
except ImportError as e:
    print(f"Critical Error: Failed to import necessary project modules: {e}", file=sys.stderr)
    print("Please ensure the project structure is correct and PYTHONPATH is set if necessary.", file=sys.stderr)
    sys.exit(1)

# Try importing colorlog, optional
try:
    import colorlog
    COLORLOG_AVAILABLE = True
except ImportError:
    COLORLOG_AVAILABLE = False

logger = logging.getLogger("BattleshipRL") # Get root logger for the project


def setup_logging(log_level_str: str = 'INFO', log_file: Optional[str] = 'logs/training.log') -> None:
    """
    Configures the root logger for the application.

    Sets up both console and file handlers. Console logs can be colored if
    'colorlog' is installed.

    Args:
        log_level_str (str): The logging level as a string (e.g., 'DEBUG', 'INFO').
        log_file (Optional[str]): Path to the log file. If None, file logging is disabled.
                                   Directory will be created if it doesn't exist.
    """
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    # --- Console Handler ---
    if COLORLOG_AVAILABLE:
        print("Info: colorlog found, enabling colored console logging.")
        # Specific format for colorlog
        c_format = '%(log_color)s' + log_format
        formatter = colorlog.ColoredFormatter(
            c_format,
            datefmt=date_format,
            log_colors={
                'DEBUG':    'cyan',
                'INFO':     'green',
                'WARNING':  'yellow',
                'ERROR':    'red',
                'CRITICAL': 'red,bg_white',
            },
            secondary_log_colors={},
            style='%'
        )
    else:
        print("Info: colorlog not found, using standard console logging.")
        formatter = logging.Formatter(log_format, datefmt=date_format)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # --- File Handler ---
    file_handler = None
    if log_file:
        log_file = os.path.join(PROJECT_ROOT, log_file) # Ensure path is relative to project root
        log_dir = os.path.dirname(log_file)
        try:
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
                print(f"Info: Created log directory: {log_dir}")

            file_handler = logging.FileHandler(log_file, mode='a') # Append mode
            file_formatter = logging.Formatter(log_format, datefmt=date_format)
            file_handler.setFormatter(file_formatter)
            print(f"Info: Logging to file: {log_file}")
        except OSError as e:
            print(f"Warning: Could not create log directory or file: {e}. File logging disabled.", file=sys.stderr)
            file_handler = None

    # Configure root logger
    root_logger = logging.getLogger() # Get the root logger
    root_logger.setLevel(log_level)
    root_logger.handlers.clear() # Remove any default handlers
    root_logger.addHandler(console_handler)
    if file_handler:
        root_logger.addHandler(file_handler)

    logger.info(f"Logging configured with level {log_level_str}.")
    # Suppress overly verbose logs from libraries if needed
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('pygame').setLevel(logging.INFO) # Or WARNING


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the training script."""
    parser = argparse.ArgumentParser(description="Train RL agents to play Battleship.")

    # --- General Training ---
    parser.add_argument('--episodes', type=int, default=10000, help='Number of training episodes to run.')
    parser.add_argument('--visualize-every', type=int, default=100, help='Frequency of game visualization (every N episodes). Set to 0 to disable.')
    parser.add_argument('--save-every', type=int, default=500, help='Frequency of saving agent models (every N episodes). Set to 0 to disable.')

    # --- Logging ---
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Set the logging verbosity.')
    parser.add_argument('--log-file', type=str, default='logs/training.log', help='Path to save detailed logs, relative to project root.')

    # --- File Paths ---
    parser.add_argument('--q-agent-path', type=str, default='models/q_learning_agent.pkl', help='File path for Q-learning agent Q-table, relative to project root.')
    parser.add_argument('--dqn-agent-path', type=str, default='models/dqn_agent.pth', help='File path for DQN agent model weights, relative to project root.')
    parser.add_argument('--metrics-path', type=str, default='metrics/training_metrics.json', help='File path for saving training metrics, relative to project root.')

    # --- Agent Hyperparameters ---
    # Shared Epsilon
    parser.add_argument('--epsilon-start', type=float, default=1.0, help='Initial exploration rate (epsilon).')
    parser.add_argument('--epsilon-decay', type=float, default=0.9995, help='Multiplicative decay factor for epsilon.')
    parser.add_argument('--epsilon-min-q', type=float, default=0.01, help='Minimum epsilon value for Q-learning agent.')
    parser.add_argument('--epsilon-min-dqn', type=float, default=0.05, help='Minimum epsilon value for DQN agent.')

    # Q-Learning Specific
    parser.add_argument('--q-learning-rate', type=float, default=0.1, help='Learning rate (alpha) for Q-learning agent.')
    parser.add_argument('--q-discount', type=float, default=0.95, help='Discount factor (gamma) for Q-learning agent.')

    # DQN Specific
    parser.add_argument('--dqn-learning-rate', type=float, default=0.0005, help='Learning rate for DQN agent optimizer.')
    parser.add_argument('--dqn-gamma', type=float, default=0.99, help='Discount factor (gamma) for DQN agent.')
    parser.add_argument('--dqn-buffer-size', type=int, default=10000, help='Capacity of the replay buffer for DQN.')
    parser.add_argument('--dqn-batch-size', type=int, default=64, help='Batch size for sampling from replay buffer in DQN.')
    parser.add_argument('--dqn-target-update', type=int, default=10, help='Frequency (in learning steps) for hard target network updates in DQN.')
    parser.add_argument('--dqn-tau', type=float, default=0.005, help='Interpolation factor for soft target updates in DQN.')
    parser.add_argument('--dqn-use-soft-update', action=argparse.BooleanOptionalAction, default=True, help='Use soft updates instead of hard updates for DQN target network.')
    parser.add_argument('--use-gpu', action='store_true', help='Flag to enable CUDA (GPU) usage for DQN if available.')

    args = parser.parse_args()
    print(f"Info: Parsed command line arguments: {vars(args)}")
    return args


def ensure_dirs_exist(args: argparse.Namespace) -> None:
    """Creates directories needed for saving models and metrics if they don't exist."""
    logger.info("Ensuring required directories exist...")
    paths_to_check = [args.q_agent_path, args.dqn_agent_path, args.metrics_path]
    for file_path in paths_to_check:
        full_path = os.path.join(PROJECT_ROOT, file_path)
        directory = os.path.dirname(full_path)
        if directory and not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
                logger.info(f"Created directory: {directory}")
            except OSError as e:
                logger.error(f"Failed to create directory {directory}: {e}", exc_info=True)
                # Decide if this is critical? For models/metrics, probably yes.
                print(f"Error: Could not create essential directory {directory}. Exiting.", file=sys.stderr)
                sys.exit(1)


def build_agent_configs(args: argparse.Namespace) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Builds configuration dictionaries for the Q-learning (Player 0) and DQN (Player 1) agents.

    Args:
        args: Parsed command-line arguments.

    Returns:
        A tuple containing two dictionaries: (config_agent_0, config_agent_1)
    """
    logger.info("Building agent configurations...")

    # --- Configuration for Agent 0 (Q-Learning) ---
    agent0_config = {
        'type': 'q_learning', # Explicitly define type
        # board_size is determined by the game instance, passed by Trainer
        'learning_rate': args.q_learning_rate,
        'discount_factor': args.q_discount,
        'epsilon_start': args.epsilon_start,
        'epsilon_decay': args.epsilon_decay,
        'min_epsilon': args.epsilon_min_q,
        # q_table_path is passed by Trainer using args.q_agent_path
    }
    logger.debug(f"Agent 0 (Q-Learning) Config: {agent0_config}")

    # --- Configuration for Agent 1 (DQN) ---
    # Determine device for DQN
    device = None
    if args.use_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info("CUDA (GPU) is available and selected for DQN.")
            else:
                device = torch.device("cpu")
                logger.warning("GPU usage requested (--use-gpu), but CUDA is not available. Using CPU for DQN.")
        except ImportError:
             device = None # Use default device detection in agent if torch fails here
             logger.warning("Cannot import torch to check GPU. DQN agent will attempt default device detection.")
    else:
        logger.info("Using CPU for DQN.")


    agent1_config = {
        'type': 'dqn', # Explicitly define type
        # board_size is passed by Trainer
        'learning_rate': args.dqn_learning_rate,
        'gamma': args.dqn_gamma,
        'epsilon_start': args.epsilon_start,
        'epsilon_decay': args.epsilon_decay,
        'epsilon_min': args.epsilon_min_dqn,
        'buffer_size': args.dqn_buffer_size,
        'batch_size': args.dqn_batch_size,
        'target_update_freq': args.dqn_target_update,
        'tau': args.dqn_tau,
        'use_soft_update': args.dqn_use_soft_update,
        'device': device, # Pass the determined device (can be None)
        # model_path is passed by Trainer using args.dqn_agent_path
    }
    logger.debug(f"Agent 1 (DQN) Config: {agent1_config}")

    return agent0_config, agent1_config

def print_training_summary(start_time: float, end_time: float, episodes_run: int, total_episodes: int) -> None:
    """Logs a summary of the training session."""
    duration = end_time - start_time
    duration_str = time.strftime("%H hours, %M minutes, %S seconds", time.gmtime(duration))
    logger.info("=" * 60)
    logger.info(" Training Summary")
    logger.info("=" * 60)
    if episodes_run < total_episodes:
        logger.warning(f"Training stopped early after {episodes_run}/{total_episodes} episodes.")
    else:
        logger.info(f"Training completed for {episodes_run} episodes.")
    logger.info(f"Total Training Duration: {duration_str} ({duration:.2f} seconds)")
    logger.info("Check log file and metrics file/plots for detailed results.")
    logger.info("=" * 60)


def main():
    """Main function to orchestrate the training process."""
    start_time = time.time()

    # 1. Parse Arguments
    args = parse_arguments()

    # 2. Setup Logging (early, so everything gets logged)
    setup_logging(args.log_level, args.log_file)
    logger.info("Starting Battleship RL Training...")
    logger.info(f"Project Root: {PROJECT_ROOT}")
    logger.info(f"Full Command: {' '.join(sys.argv)}") # Log the exact command used

    # 3. Ensure Directories Exist
    ensure_dirs_exist(args)

    # 4. Build Agent Configurations
    try:
        agent0_cfg, agent1_cfg = build_agent_configs(args)
    except Exception as e:
        logger.exception("Failed to build agent configurations.", exc_info=True)
        print(f"Error: Failed to configure agents: {e}. Exiting.", file=sys.stderr)
        sys.exit(1)

    # 5. Initialize Trainer
    logger.info("Initializing the Trainer...")
    try:
        trainer = Trainer(
            num_episodes=args.episodes,
            visualize_every_n=args.visualize_every,
            save_every_n=args.save_every,
            agent0_config=agent0_cfg,
            agent1_config=agent1_cfg,
            q_agent_save_path=os.path.join(PROJECT_ROOT, args.q_agent_path), # Use absolute paths
            dqn_agent_save_path=os.path.join(PROJECT_ROOT, args.dqn_agent_path),
            metrics_save_path=os.path.join(PROJECT_ROOT, args.metrics_path)
        )
        logger.info("Trainer initialized successfully.")
    except Exception as e:
        logger.exception("Failed to initialize the Trainer.", exc_info=True)
        print(f"Error: Failed to initialize Trainer: {e}. Exiting.", file=sys.stderr)
        sys.exit(1)

    # 6. Run Training
    episodes_completed = 0
    try:
        logger.info("Starting training loop...")
        trainer.train() # This method now handles the loop and periodic reporting/saving
        # Trainer's train method should log completion or abortion details.
        # We can access metrics to find out how many episodes actually ran if needed.
        episodes_completed = len(trainer.metrics.episode_data) # Get actual count from metrics

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user (KeyboardInterrupt).")
        print("\nTraining interrupted by user. Saving progress...", file=sys.stderr)
        # Trainer should ideally handle saving on interruption, but we can force save here too.
        if trainer:
             logger.info("Attempting to save models and metrics after interruption...")
             # Check agent types before saving to avoid errors if one agent type wasn't created
             if isinstance(trainer.agents.get(0), QLearningAgent) or isinstance(trainer.agents.get(1), QLearningAgent):
                q_agent = trainer.agents[0] if isinstance(trainer.agents.get(0), QLearningAgent) else trainer.agents.get(1)
                if q_agent: q_agent.save_q_table()
             if isinstance(trainer.agents.get(0), DQNAgent) or isinstance(trainer.agents.get(1), DQNAgent):
                dqn_agent = trainer.agents[0] if isinstance(trainer.agents.get(0), DQNAgent) else trainer.agents.get(1)
                if dqn_agent: dqn_agent.save_model()
             trainer.metrics.save_metrics()
             logger.info("Progress saved.")
        episodes_completed = len(trainer.metrics.episode_data) if trainer else 0


    except Exception as e:
        logger.exception("An unexpected error occurred during training.", exc_info=True)
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
        episodes_completed = len(trainer.metrics.episode_data) if trainer else 0
        # Consider saving progress here as well? Depends on the error.

    finally:
        # 7. Print Summary
        end_time = time.time()
        print_training_summary(start_time, end_time, episodes_completed, args.episodes)
        # Ensure Pygame is quit if visualization was used and not closed properly
        if trainer and trainer.renderer:
            trainer.renderer.close() # Safe to call even if already closed


if __name__ == "__main__":
    main()