# Generated with write_code_to_file
# /home/myuser/apps/battleshiprl/utils/metrics.py

import json
import os
import time
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple

# Try importing matplotlib for plotting, but make it optional
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
    logger_mpl = logging.getLogger('matplotlib')
    logger_mpl.setLevel(logging.WARNING) # Suppress matplotlib debug logs
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# Set up logging for this module
logger = logging.getLogger(__name__)
# Basic config if run standalone or not configured elsewhere:
if not logger.hasHandlers() or not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class TrainingMetrics:
    """
    Collects, analyzes, visualizes, and saves metrics during the RL training process.
    """

    DEFAULT_WINDOW_SIZE = 100 # Window size for moving averages

    def __init__(self, agent_types: Dict[int, str], save_path: Optional[str] = 'training_metrics.json'):
        """
        Initializes the TrainingMetrics collector.

        Args:
            agent_types (Dict[int, str]): Dictionary mapping player ID (0 or 1) to agent type name (e.g., 'QLearningAgent').
            save_path (Optional[str]): Path to save/load the metrics data as JSON. If None, saving/loading is disabled.
        """
        logger.info(f"Initializing TrainingMetrics. Agent Types: {agent_types}, Save Path: {save_path}")
        self.agent_types: Dict[int, str] = agent_types
        self.save_path: Optional[str] = save_path
        # List to store data for each episode. Each entry is a dictionary.
        self.episode_data: List[Dict[str, Any]] = []

        # Load existing metrics if path is valid and file exists
        if self.save_path:
            self.load_metrics()
        else:
            logger.warning("No save path provided for metrics. Data will not be persisted.")

        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib not found. Plotting functionality will be disabled.")

    def record_episode(self,
                         episode_num: int,
                         winner: Optional[int],
                         turns: int,
                         duration: float,
                         epsilon0: Optional[float] = None, # Make epsilons optional if an agent doesn't have it
                         epsilon1: Optional[float] = None) -> None:
        """
        Records the results of a single training episode.

        Args:
            episode_num (int): The episode number (starting from 1).
            winner (Optional[int]): The ID of the winning player (0 or 1), or None if aborted/draw.
            turns (int): The number of turns the episode lasted.
            duration (float): The time duration of the episode in seconds.
            epsilon0 (Optional[float]): Exploration rate of agent 0 at the end of the episode.
            epsilon1 (Optional[float]): Exploration rate of agent 1 at the end of the episode.
        """
        timestamp = time.time()
        episode_record = {
            "episode": episode_num,
            "timestamp": timestamp,
            "winner": winner,
            "turns": turns,
            "duration_sec": duration,
            "epsilon0": epsilon0,
            "epsilon1": epsilon1,
        }
        self.episode_data.append(episode_record)
        logger.debug(f"Recorded episode {episode_num}: Winner={winner}, Turns={turns}, Duration={duration:.2f}s, Eps={epsilon0:.4f}/{epsilon1:.4f}")

        # Optional: Auto-save periodically based on episode count or time?
        # For now, saving is triggered externally by the Trainer.

    def _calculate_sma(self, data: List[float], window: int) -> List[float]:
        """Calculates the Simple Moving Average (SMA) for a list of data."""
        if not data or window <= 0:
            return []
        if len(data) < window:
            # For initial phase, calculate cumulative average until window is full
            return [np.mean(data[:i+1]) for i in range(len(data))]
        else:
            # Use numpy's convolution for efficient SMA calculation
            # Note: 'valid' mode starts after the first window is full. Pad start for full length.
            initial_cumulative = [np.mean(data[:i+1]) for i in range(window - 1)]
            weights = np.repeat(1.0, window) / window
            sma = np.convolve(data, weights, 'valid')
            return initial_cumulative + list(sma)

    def get_win_rates(self, window: int = DEFAULT_WINDOW_SIZE) -> Dict[str, List[float]]:
        """
        Calculates the win rate for each player over a moving window.

        Args:
            window (int): The number of episodes to include in the moving average window.

        Returns:
            Dict[str, List[float]]: Dictionary containing lists of win rates:
                                    'episodes': Episode numbers corresponding to the rates.
                                    'player0_win_rate': Moving average win rate for player 0.
                                    'player1_win_rate': Moving average win rate for player 1.
        """
        if not self.episode_data:
            return {"episodes": [], "player0_win_rate": [], "player1_win_rate": []}

        episodes = [e['episode'] for e in self.episode_data]
        wins0 = [1.0 if e['winner'] == 0 else 0.0 for e in self.episode_data]
        wins1 = [1.0 if e['winner'] == 1 else 0.0 for e in self.episode_data]

        # Use numpy for efficient rolling mean calculation if available and desired,
        # otherwise use custom SMA function.
        win_rate0 = self._calculate_sma(wins0, window)
        win_rate1 = self._calculate_sma(wins1, window)

        return {
            "episodes": episodes,
            "player0_win_rate": win_rate0,
            "player1_win_rate": win_rate1
        }

    def get_average_turns(self, window: int = DEFAULT_WINDOW_SIZE) -> Dict[str, List[float]]:
        """
        Calculates the average number of turns per episode over a moving window.

        Args:
            window (int): The number of episodes to include in the moving average window.

        Returns:
            Dict[str, List[float]]: Dictionary containing:
                                    'episodes': Episode numbers.
                                    'average_turns': Moving average of turns per episode.
        """
        if not self.episode_data:
            return {"episodes": [], "average_turns": []}

        episodes = [e['episode'] for e in self.episode_data]
        turns = [float(e['turns']) for e in self.episode_data]

        avg_turns = self._calculate_sma(turns, window)

        return {"episodes": episodes, "average_turns": avg_turns}

    def get_epsilon_trends(self) -> Dict[str, List[Optional[float]]]:
        """
        Extracts the epsilon values recorded for each agent over all episodes.

        Returns:
            Dict[str, List[Optional[float]]]: Dictionary containing:
                                              'episodes': Episode numbers.
                                              'epsilon0': List of epsilon values for agent 0.
                                              'epsilon1': List of epsilon values for agent 1.
        """
        if not self.episode_data:
            return {"episodes": [], "epsilon0": [], "epsilon1": []}

        episodes = [e['episode'] for e in self.episode_data]
        eps0 = [e.get('epsilon0') for e in self.episode_data] # Use .get in case not present
        eps1 = [e.get('epsilon1') for e in self.episode_data]

        return {"episodes": episodes, "epsilon0": eps0, "epsilon1": eps1}

    def plot_metrics(self, window: int = DEFAULT_WINDOW_SIZE, save_fig_path: Optional[str] = "training_metrics_plot.png"):
        """
        Generates and saves plots for key training metrics using matplotlib (if available).

        Args:
            window (int): The window size for calculating moving averages in plots.
            save_fig_path (Optional[str]): Path to save the generated plot image. If None, plot is shown but not saved.
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Cannot plot metrics: matplotlib is not installed.")
            return
        if not self.episode_data:
            logger.warning("Cannot plot metrics: No episode data recorded.")
            return

        logger.info(f"Generating metrics plots with window size {window}...")

        try:
            win_rates_data = self.get_win_rates(window)
            avg_turns_data = self.get_average_turns(window)
            epsilon_data = self.get_epsilon_trends()

            # Determine number of plots needed (usually 3)
            num_plots = 3
            fig, axes = plt.subplots(num_plots, 1, figsize=(10, 5 * num_plots), sharex=True)
            fig.suptitle('Training Metrics Over Time', fontsize=16)

            # Consistent episode list (use the longest one if they differ, though they shouldn't)
            episodes = win_rates_data['episodes']

            # --- Plot 1: Win Rates ---
            ax1 = axes[0]
            label0 = f"Player 0 ({self.agent_types.get(0, 'Unknown')})"
            label1 = f"Player 1 ({self.agent_types.get(1, 'Unknown')})"
            ax1.plot(episodes, win_rates_data['player0_win_rate'], label=label0, color='green')
            ax1.plot(episodes, win_rates_data['player1_win_rate'], label=label1, color='red')
            ax1.set_ylabel(f'Win Rate (Avg over {window} eps)')
            ax1.set_title('Agent Win Rates')
            ax1.legend()
            ax1.grid(True)
            ax1.set_ylim(0, 1.05) # Win rate is between 0 and 1

            # --- Plot 2: Average Game Length ---
            ax2 = axes[1]
            ax2.plot(episodes, avg_turns_data['average_turns'], label='Average Turns', color='blue')
            ax2.set_ylabel(f'Turns (Avg over {window} eps)')
            ax2.set_title('Average Game Length (Turns)')
            ax2.legend()
            ax2.grid(True)
            # Optional: Set y-limit based on data range?
            min_turns = min(avg_turns_data['average_turns']) if avg_turns_data['average_turns'] else 0
            max_turns = max(avg_turns_data['average_turns']) if avg_turns_data['average_turns'] else 100 # Default max
            ax2.set_ylim(max(0, min_turns - 10), max_turns + 10)


            # --- Plot 3: Epsilon Decay ---
            ax3 = axes[2]
            eps0 = [e for e in epsilon_data['epsilon0'] if e is not None]
            eps1 = [e for e in epsilon_data['epsilon1'] if e is not None]
            eps_episodes = episodes # Assumes epsilon tracked for all episodes
            # Filter episodes if epsilon wasn't always tracked (adjust eps_episodes)
            # For simplicity now assume tracked for all or handle potential length mismatch in plotting

            if eps0:
                 ax3.plot(eps_episodes[:len(eps0)], eps0, label=f'{label0} Epsilon', color='lightgreen', linestyle='--')
            if eps1:
                 ax3.plot(eps_episodes[:len(eps1)], eps1, label=f'{label1} Epsilon', color='lightcoral', linestyle='--')

            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Epsilon Value')
            ax3.set_title('Exploration Rate (Epsilon) Decay')
            if eps0 or eps1: # Only show legend if there's data
                 ax3.legend()
            ax3.grid(True)
            ax3.set_ylim(0, 1.05) # Epsilon is between 0 and 1

            # Adjust layout and save/show
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

            if save_fig_path:
                 # Ensure directory exists
                 fig_dir = os.path.dirname(save_fig_path)
                 if fig_dir and not os.path.exists(fig_dir):
                     os.makedirs(fig_dir, exist_ok=True)
                 plt.savefig(save_fig_path)
                 logger.info(f"Metrics plot saved to {save_fig_path}")
            else:
                 # plt.show() # Display plot interactively if not saving (can block execution)
                 # Let's avoid blocking during training, maybe just log that plotting happened.
                 logger.info("Metrics plot generated (matplotlib window).")

            plt.close(fig) # Close the figure to free memory

        except Exception as e:
            logger.error(f"Error generating metrics plot: {e}", exc_info=True)


    def save_metrics(self, file_path: Optional[str] = None) -> None:
        """
        Saves the recorded episode data to a JSON file.

        Args:
            file_path (Optional[str]): The path to save the JSON file. If None, uses the
                                       `save_path` provided during initialization.
        """
        path = file_path or self.save_path
        if not path:
            logger.warning("Cannot save metrics: No save path specified.")
            return
        if not self.episode_data:
            logger.info("No metrics data to save.")
            return

        # Ensure the directory exists
        try:
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)
        except OSError as e:
            logger.error(f"Error creating directory {directory} for metrics file: {e}")
            return

        try:
            with open(path, 'w') as f:
                json.dump(self.episode_data, f, indent=4)
            logger.info(f"Metrics data ({len(self.episode_data)} episodes) saved successfully to {path}")
        except (IOError, TypeError, Exception) as e:
            logger.error(f"Error saving metrics data to {path}: {e}")

    def load_metrics(self, file_path: Optional[str] = None) -> None:
        """
        Loads episode data from a JSON file. Appends to existing data if any.

        Args:
            file_path (Optional[str]): The path to load the JSON file from. If None, uses the
                                       `save_path` provided during initialization.
        """
        path = file_path or self.save_path
        if not path:
            logger.warning("Cannot load metrics: No load path specified.")
            return

        if not os.path.exists(path) or os.path.getsize(path) == 0:
            logger.info(f"Metrics file not found or is empty at {path}. Starting with no historical data.")
            self.episode_data = []
            return

        try:
            with open(path, 'r') as f:
                loaded_data = json.load(f)

            if isinstance(loaded_data, list):
                # Basic check: ensure items in list look like episode records (dicts with 'episode' key)
                if all(isinstance(item, dict) and 'episode' in item for item in loaded_data):
                    # Decide whether to append or replace. Let's replace for simplicity,
                    # assuming load happens only at init. If resuming, appending might be needed.
                    # For now, replace.
                    self.episode_data = loaded_data
                    logger.info(f"Metrics data ({len(self.episode_data)} episodes) loaded successfully from {path}")
                else:
                    logger.error(f"Invalid format in metrics file {path}. Expected list of episode dicts. Resetting metrics.")
                    self.episode_data = []
            else:
                logger.error(f"Invalid format in metrics file {path}. Expected a JSON list. Resetting metrics.")
                self.episode_data = []

        except (IOError, json.JSONDecodeError, TypeError, Exception) as e:
            logger.error(f"Error loading metrics data from {path}: {e}. Resetting metrics.")
            self.episode_data = []


    def print_progress(self, current_episode: int, total_episodes: int, start_time: float, interval: int = 100):
        """
        Prints a progress update to the console, including estimated time remaining
        and recent performance metrics.

        Args:
            current_episode (int): The number of the episode just completed.
            total_episodes (int): The total number of episodes planned for training.
            start_time (float): The timestamp (from time.time()) when training started.
            interval (int): The window size for calculating recent performance (e.g., last 100 episodes).
        """
        elapsed_time = time.time() - start_time
        episodes_remaining = total_episodes - current_episode
        avg_time_per_episode = elapsed_time / current_episode if current_episode > 0 else 0
        estimated_remaining_time = episodes_remaining * avg_time_per_episode if avg_time_per_episode > 0 else 0

        # Format time nicely
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        eta_str = time.strftime("%H:%M:%S", time.gmtime(estimated_remaining_time)) if estimated_remaining_time > 0 else "--:--:--"

        # Calculate recent performance (over the last 'interval' episodes)
        recent_data = self.episode_data[-interval:]
        recent_wins0 = sum(1 for e in recent_data if e['winner'] == 0)
        recent_wins1 = sum(1 for e in recent_data if e['winner'] == 1)
        recent_turns = [e['turns'] for e in recent_data]
        avg_recent_turns = np.mean(recent_turns) if recent_turns else 0
        num_recent = len(recent_data)
        recent_win_rate0 = recent_wins0 / num_recent if num_recent > 0 else 0
        recent_win_rate1 = recent_wins1 / num_recent if num_recent > 0 else 0

        # Get current epsilon values
        eps0 = self.episode_data[-1].get('epsilon0', -1.0) if self.episode_data else -1.0
        eps1 = self.episode_data[-1].get('epsilon1', -1.0) if self.episode_data else -1.0

        # Build progress string
        progress_pct = (current_episode / total_episodes) * 100
        progress_str = (
            f"Episode {current_episode}/{total_episodes} [{progress_pct:.1f}%] | "
            f"Elapsed: {elapsed_str} | ETA: {eta_str} | "
            f"Last {interval} Avg Turns: {avg_recent_turns:.1f} | "
            f"Last {interval} WR (P0/P1): {recent_win_rate0*100:.1f}% / {recent_win_rate1*100:.1f}% | "
            f"Eps (P0/P1): {eps0:.4f}/{eps1:.4f}"
        )

        print(progress_str)  # Use print for direct console output during training

# Example Usage (for testing the metrics module directly)
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG) # Show debug messages for testing
    logger.info("Starting TrainingMetrics example usage...")

    test_save_path = "temp_test_metrics.json"
    # Clean up previous test file
    if os.path.exists(test_save_path):
        os.remove(test_save_path)

    agent_types_test = {0: "QLearner", 1: "DQN"}
    metrics = TrainingMetrics(agent_types=agent_types_test, save_path=test_save_path)

    # Simulate recording some episodes
    start_train_time = time.time()
    num_test_eps = 250
    current_eps0 = 1.0
    current_eps1 = 1.0
    eps_decay = 0.99

    for i in range(1, num_test_eps + 1):
        winner = random.choice([0, 1, None]) # Simulate random winner or None
        turns = random.randint(50, 150)
        duration = random.uniform(0.5, 2.0)
        current_eps0 = max(0.01, current_eps0 * eps_decay)
        current_eps1 = max(0.05, current_eps1 * eps_decay * 0.99) # Slightly different decay for testing

        metrics.record_episode(
            episode_num=i,
            winner=winner,
            turns=turns,
            duration=duration,
            epsilon0=current_eps0,
            epsilon1=current_eps1
        )
        if i % 50 == 0:
            metrics.print_progress(i, num_test_eps, start_train_time, interval=50)


    # --- Test calculations ---
    print("\n--- Testing Metric Calculations ---")
    win_rates = metrics.get_win_rates(window=50)
    avg_turns = metrics.get_average_turns(window=50)
    eps_trends = metrics.get_epsilon_trends()

    print(f"Win Rates (Player 0, last value): {win_rates['player0_win_rate'][-1]:.3f}" if win_rates['player0_win_rate'] else "N/A")
    print(f"Avg Turns (last value): {avg_turns['average_turns'][-1]:.1f}" if avg_turns['average_turns'] else "N/A")
    print(f"Epsilon 0 (last value): {eps_trends['epsilon0'][-1]:.4f}" if eps_trends['epsilon0'] else "N/A")


    # --- Test Plotting ---
    print("\n--- Testing Plotting ---")
    metrics.plot_metrics(window=50, save_fig_path="temp_test_metrics_plot.png")
    if MATPLOTLIB_AVAILABLE:
         print("Plotting attempted. Check for 'temp_test_metrics_plot.png'")
         # Clean up plot file
         if os.path.exists("temp_test_metrics_plot.png"):
             os.remove("temp_test_metrics_plot.png")
    else:
         print("Plotting skipped as matplotlib is not available.")

    # --- Test Saving ---
    print("\n--- Testing Saving ---")
    metrics.save_metrics()

    # --- Test Loading ---
    print("\n--- Testing Loading ---")
    # Create a new instance and load
    metrics_loaded = TrainingMetrics(agent_types=agent_types_test, save_path=test_save_path)

    print(f"Original episode count: {len(metrics.episode_data)}")
    print(f"Loaded episode count: {len(metrics_loaded.episode_data)}")
    assert len(metrics.episode_data) == len(metrics_loaded.episode_data), "Loaded data count mismatch!"
    if metrics.episode_data and metrics_loaded.episode_data:
        assert metrics.episode_data[-1] == metrics_loaded.episode_data[-1], "Loaded data content mismatch!"
    print("Loading test successful.")

    # --- Cleanup ---
    if os.path.exists(test_save_path):
        os.remove(test_save_path)
        print(f"Cleaned up temporary file: {test_save_path}")

    logger.info("Finished TrainingMetrics example usage.")

# Potential Improvements:
# - More sophisticated statistics (e.g., standard deviation, win streaks).
# - Options for different moving average types (e.g., exponential moving average).
# - More flexible plotting options (choosing which plots, custom styling).
# - Integration with logging frameworks for more structured output.
# - Performance optimization for very large datasets (e.g., using pandas if acceptable).
# - Handling of 'None' winner states more explicitly in win rate calculation if draws are possible.