# Generated with write_code_to_file
# /home/myuser/apps/battleshiprl/battleship/game.py

import numpy as np
import random
import logging
from typing import Dict, Tuple, List, Optional, Any, Set

# Correct import path for Ship module based on the file tree
try:
    from battleship.ship import Ship, SHIPS
except ImportError:
    # Handle potential import errors, perhaps if run standalone or path issues.
    # This might involve adjusting sys.path depending on execution context.
    logging.error("Failed to import Ship class from battleship.ship. Check PYTHONPATH.")
    # Fallback definition of SHIPS if import fails, for basic functionality.
    SHIPS = {
        "Carrier": 5,
        "Battleship": 4,
        "Cruiser": 3,
        "Submarine": 3,
        "Destroyer": 2,
    }
    # If Ship class is critical and import fails, we might need to exit or raise error.
    # For now, let it proceed but log the error. Define a dummy Ship if needed for structure.
    # class Ship: def __init__(self, *args): pass # Dummy class

# --- Constants ---
BOARD_SIZE = 10
BOARD_EMPTY = 0      # Cell state: empty water
BOARD_SHIP = 1       # Cell state: contains an unhit part of a ship (on player's own board)
BOARD_HIT = 2        # Cell state: contains a hit part of a ship
BOARD_MISS = 3       # Cell state: empty water that has been fired upon

# Set up logging
logger = logging.getLogger(__name__)
# Basic config if no handlers are configured by the root logger (e.g., in main.py)
if not logger.hasHandlers():
     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class BattleshipGame:
    """
    Implements the core logic for the Battleship game.
    Manages board state, ship placement, turns, and win conditions.
    """

    def __init__(self, board_size: int = BOARD_SIZE):
        """
        Initializes the Battleship game environment.

        Args:
            board_size (int): The dimension of the square game board.
        """
        logger.info(f"Initializing Battleship game with board size {board_size}x{board_size}")
        self.board_size = board_size

        # Initialize boards:
        # 'boards' stores the ground truth for each player (where their ships are)
        # 'opponent_views' stores what each player can see of the opponent's board (hits/misses)
        self.boards: Dict[int, np.ndarray] = {
            0: np.full((board_size, board_size), BOARD_EMPTY, dtype=int),
            1: np.full((board_size, board_size), BOARD_EMPTY, dtype=int)
        }
        self.opponent_views: Dict[int, np.ndarray] = {
            0: np.full((board_size, board_size), BOARD_EMPTY, dtype=int),
            1: np.full((board_size, board_size), BOARD_EMPTY, dtype=int)
        }

        # Ships for each player
        self.ships: Dict[int, List[Ship]] = {0: [], 1: []}

        # Game state variables
        self.turns_taken: int = 0
        # Start with player 0 or random? Let's start with player 0 consistently.
        self.current_player: int = 0
        self.game_over: bool = False
        self.winner: Optional[int] = None

        # Place ships for both players
        self._setup_game()
        logger.info("Game initialized and ships placed.")

    def _setup_game(self):
        """Places ships for both players on their respective boards."""
        logger.debug("Setting up game - placing ships for both players.")
        # Handle potential errors during placement (though `place_ships` should be robust)
        try:
            self.place_ships(0)
            logger.debug("Player 0 ships placed.")
            self.place_ships(1)
            logger.debug("Player 1 ships placed.")
        except Exception as e:
             logger.error(f"Error during initial ship placement: {e}", exc_info=True)
             # This is critical, the game cannot start. Raise or handle appropriately.
             raise RuntimeError("Failed to place ships during game setup.") from e


    def is_valid_placement(self, player_id: int, coordinates: List[Tuple[int, int]]) -> bool:
        """
        Checks if a list of coordinates is a valid placement location for a ship
        on the specified player's board.

        Args:
            player_id (int): The player ID (0 or 1) whose board to check.
            coordinates (List[Tuple[int, int]]): List of (row, col) coordinates for the ship.

        Returns:
            bool: True if the placement is valid, False otherwise.
        """
        for r, c in coordinates:
            # Check 1: Within board bounds
            if not (0 <= r < self.board_size and 0 <= c < self.board_size):
                logger.debug(f"Invalid placement: Coordinate {(r, c)} is out of bounds.")
                return False
            # Check 2: Not overlapping with existing ships
            if self.boards[player_id][r, c] != BOARD_EMPTY:
                # Allows placement over misses, but not over ships or hits
                # Let's restrict: cannot place on *anything* but empty
                 logger.debug(f"Invalid placement: Coordinate {(r, c)} is not empty (contains {self.boards[player_id][r, c]}).")
                 return False # Cannot place on squares that are not empty

        # All checks passed
        return True

    def place_ships(self, player_id: int):
        """
        Randomly places all standard ships for the given player.
        Ensures ships are within bounds, do not overlap, and are placed
        horizontally or vertically.

        Args:
            player_id (int): The player ID (0 or 1) for whom to place ships.

        Raises:
            RuntimeError: If it fails to place a ship after many attempts (potential bug or impossible layout).
        """
        logger.info(f"Attempting to randomly place ships for Player {player_id}.")
        self.ships[player_id] = [] # Clear existing ships if any
        # Reset the player's board section related to ships before placing new ones
        self.boards[player_id][self.boards[player_id] == BOARD_SHIP] = BOARD_EMPTY
        self.boards[player_id][self.boards[player_id] == BOARD_HIT] = BOARD_EMPTY # Also clear hits if resetting


        max_attempts_per_ship = 1000 # Prevent infinite loops

        for ship_type, size in SHIPS.items():
            ship = Ship(ship_type, size)
            placed = False
            attempts = 0

            while not placed and attempts < max_attempts_per_ship:
                attempts += 1
                coordinates = []
                # Choose random orientation (0: horizontal, 1: vertical)
                orientation = random.choice([0, 1])
                # Choose random start position
                start_row = random.randint(0, self.board_size - 1)
                start_col = random.randint(0, self.board_size - 1)

                # Calculate potential coordinates
                valid_coords = True
                for i in range(size):
                    if orientation == 0: # Horizontal
                        r, c = start_row, start_col + i
                    else: # Vertical
                        r, c = start_row + i, start_col
                    coordinates.append((r, c))

                # Validate the calculated coordinates
                if self.is_valid_placement(player_id, coordinates):
                    # Place the ship
                    ship.place(coordinates)
                    self.ships[player_id].append(ship)
                    # Update the board state
                    for r, c in coordinates:
                        self.boards[player_id][r, c] = BOARD_SHIP
                    placed = True
                    logger.debug(f"Placed {ship_type} for Player {player_id} at {coordinates} (Attempt {attempts}).")
                # else: logger.debug(f"Attempt {attempts} for {ship_type} failed.") # Can be very verbose


            if not placed:
                 logger.error(f"Failed to place ship {ship_type} (Size: {size}) for Player {player_id} after {max_attempts_per_ship} attempts.")
                 raise RuntimeError(f"Could not place ship {ship_type} for Player {player_id}. Check board size and ship configuration.")

        logger.info(f"All ships placed successfully for Player {player_id}.")


    def fire(self, coordinate: Tuple[int, int]) -> Dict[str, Any]:
        """
        Processes a shot fired by the current player at the given coordinate
        on the opponent's board. Updates game state accordingly.

        Args:
            coordinate (Tuple[int, int]): The (row, col) target of the shot.

        Returns:
            Dict[str, Any]: A dictionary describing the result of the shot:
                'result': 'hit', 'miss', 'already_fired', or 'invalid'.
                'coordinate': The (row, col) tuple that was targeted.
                'sunk': The type (str) of the ship sunk, if any, otherwise None.
                'game_over': bool indicating if the game ended with this shot.
                'winner': The player ID (int) of the winner, if game_over is True, otherwise None.
        """
        row, col = coordinate
        target_player = 1 - self.current_player

        result_info = {
            'result': 'invalid',
            'coordinate': coordinate,
            'sunk': None,
            'game_over': self.game_over, # Start with current game over state
            'winner': self.winner
        }

        # 1. Validate coordinate
        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            logger.warning(f"Player {self.current_player} fired at invalid coordinate {coordinate}.")
            result_info['result'] = 'invalid'
            # Note: We might still switch player or end turn depending on rules for invalid moves.
            # For RL, penalize heavily but maybe don't switch player to let agent learn?
            # For now, let's treat invalid as a wasted turn, update turn count and switch.
            self.turns_taken += 1
            if not self.game_over: self.switch_player() # Consume turn
            return result_info

        # 2. Check if already fired upon
        if self.opponent_views[self.current_player][row, col] != BOARD_EMPTY:
            logger.warning(f"Player {self.current_player} fired at already targeted coordinate {coordinate}.")
            result_info['result'] = 'already_fired'
            # Also treat as a wasted turn
            self.turns_taken += 1
            if not self.game_over: self.switch_player() # Consume turn
            return result_info

        # 3. Determine hit or miss based on target player's actual board
        target_cell_state = self.boards[target_player][row, col]

        if target_cell_state == BOARD_SHIP:
            # --- HIT ---
            # logger.info(f"Player {self.current_player} HIT Player {target_player} at {coordinate}!")
            result_info['result'] = 'hit'
            # Update opponent's view for the current player
            self.opponent_views[self.current_player][row, col] = BOARD_HIT
            # Update the target player's actual board state
            self.boards[target_player][row, col] = BOARD_HIT

            # Check which ship was hit and if it sunk
            hit_ship_instance = None
            for ship in self.ships[target_player]:
                if ship.is_hit(coordinate): # Note: is_hit updates ship's internal state
                    hit_ship_instance = ship
                    break

            if hit_ship_instance:
                 if hit_ship_instance.is_sunk():
                     logger.info(f"Player {target_player}'s {hit_ship_instance.ship_type} SUNK!")
                     result_info['sunk'] = hit_ship_instance.ship_type
                     # Check for game over
                     if self.check_game_over(target_player):
                         logger.info(f"GAME OVER! Player {self.current_player} wins!")
                         self.game_over = True
                         self.winner = self.current_player
                         result_info['game_over'] = True
                         result_info['winner'] = self.current_player
            else:
                 # This case suggests inconsistency: board showed SHIP/HIT but no ship object registered the hit.
                 logger.error(f"Inconsistency: Hit at {coordinate} on Player {target_player}'s board, but no ship object registered the hit.")

        elif target_cell_state == BOARD_EMPTY:
            # --- MISS ---
            # logger.info(f"Player {self.current_player} missed at {coordinate}.")
            result_info['result'] = 'miss'
            # Update opponent's view for the current player
            self.opponent_views[self.current_player][row, col] = BOARD_MISS
            # Target player's actual board remains BOARD_EMPTY, but opponent view shows the miss.

        else:
             # This might happen if coordinate points to BOARD_HIT (already hit ship part)
             # which should have been caught by 'already_fired' check based on opponent_view.
             # Or if it points to BOARD_MISS (opponent view should also show this).
             # Log this potential inconsistency.
             logger.error(f"Unexpected target cell state {target_cell_state} at {coordinate} for Player {target_player}. Should have been caught earlier.")
             # Treat as 'already_fired' for robustness?
             result_info['result'] = 'already_fired' # Reclassify based on probable cause


        # 4. Update turn count and switch player (if game not over)
        self.turns_taken += 1
        if not self.game_over:
            self.switch_player()

        logger.debug(f"Fire result: {result_info}")
        return result_info


    def check_game_over(self, player_id: int) -> bool:
        """
        Checks if all ships belonging to the specified player are sunk.

        Args:
            player_id (int): The player whose ships to check.

        Returns:
            bool: True if all ships of the player are sunk, False otherwise.
        """
        if not self.ships[player_id]: # No ships placed?
             logger.warning(f"Checking game over for Player {player_id}, but they have no ships listed.")
             return False # Cannot be game over if no ships exist? Or is it? Assume not over.

        all_sunk = all(ship.is_sunk() for ship in self.ships[player_id])
        if all_sunk:
            logger.debug(f"Confirmed: All ships for Player {player_id} are sunk.")
        return all_sunk

    def switch_player(self):
        """Switches the current player ID between 0 and 1."""
        self.current_player = 1 - self.current_player
        logger.debug(f"Switched player. Current player is now {self.current_player}.")

    def get_state(self, player_id: int) -> np.ndarray:
        """
        Returns the game state representation suitable for an RL agent.
        This is typically the player's view of the opponent's board (hits/misses).

        Args:
            player_id (int): The ID of the player for whom to get the state.

        Returns:
            np.ndarray: The player's view of the opponent's board.
        """
        return self.opponent_views[player_id]

    def get_valid_actions(self, player_id: int) -> List[Tuple[int, int]]:
        """
        Returns a list of valid actions (coordinates) for the specified player.
        A valid action is a coordinate that has not yet been fired upon.

        Args:
            player_id (int): The player ID.

        Returns:
            List[Tuple[int, int]]: A list of (row, col) coordinates available to target.
        """
        valid_coords = []
        # Iterate through the player's view of the opponent's board
        view = self.opponent_views[player_id]
        # Find indices where the board value is BOARD_EMPTY using numpy.where
        empty_cells = np.where(view == BOARD_EMPTY)
        # Convert the row and column indices into a list of (row, col) tuples
        valid_coords = list(zip(empty_cells[0], empty_cells[1]))

        # Shuffle? Might help exploration but could make debugging harder. Let's keep ordered.
        # random.shuffle(valid_coords)
        logger.debug(f"Found {len(valid_coords)} valid actions for Player {player_id}.")
        return valid_coords

    def reset(self):
        """Resets the game to its initial state for a new episode."""
        logger.info("Resetting game state.")
        # Reset boards and views
        for player_id in [0, 1]:
            self.boards[player_id].fill(BOARD_EMPTY)
            self.opponent_views[player_id].fill(BOARD_EMPTY)
            self.ships[player_id] = [] # Clear ship list

        # Reset game state variables
        self.turns_taken = 0
        self.current_player = 0 # Consistent starting player
        self.game_over = False
        self.winner = None

        # Place new ships
        self._setup_game()
        logger.info("Game reset complete.")

    def render_board(self, player_id: int) -> str:
        """
        Returns a string representation of a player's primary board (where their ships are).
        Useful for debugging.

        Args:
            player_id (int): Player ID (0 or 1).

        Returns:
            str: A multi-line string representing the board.
        """
        board = self.boards[player_id]
        header = f"Player {player_id}'s Board:\n"
        col_labels = "  " + " ".join(map(str, range(self.board_size))) + "\n"
        board_str = ""
        for r in range(self.board_size):
            row_str = f"{r} "
            for c in range(self.board_size):
                cell = board[r, c]
                if cell == BOARD_EMPTY: char = '.' # Empty water
                elif cell == BOARD_SHIP: char = 'S' # Unhit Ship part
                elif cell == BOARD_HIT: char = 'X' # Hit Ship part
                elif cell == BOARD_MISS: char = 'o' # Miss (unlikely on own board unless rules change)
                else: char = '?' # Unknown state
                row_str += char + " "
            board_str += row_str.rstrip() + "\n"
        return header + col_labels + board_str

    def render_opponent_view(self, player_id: int) -> str:
        """
        Returns a string representation of what a player sees of the opponent's board.
        Useful for debugging.

        Args:
            player_id (int): Player ID (0 or 1).

        Returns:
            str: A multi-line string representing the player's view of the opponent's grid.
        """
        view = self.opponent_views[player_id]
        header = f"Player {player_id}'s View of Opponent:\n"
        col_labels = "  " + " ".join(map(str, range(self.board_size))) + "\n"
        board_str = ""
        for r in range(self.board_size):
            row_str = f"{r} "
            for c in range(self.board_size):
                cell = view[r, c]
                if cell == BOARD_EMPTY: char = '.' # Unknown / Not fired
                elif cell == BOARD_HIT: char = 'X' # Confirmed Hit
                elif cell == BOARD_MISS: char = 'o' # Confirmed Miss
                else: char = '?' # Unknown state
                row_str += char + " "
            board_str += row_str.rstrip() + "\n"
        return header + col_labels + board_str

    def __str__(self):
        """Provides a simple string representation of the game's current state."""
        status = "Game Over" if self.game_over else "In Progress"
        winner_str = f", Winner: {self.winner}" if self.game_over else ""
        return f"BattleshipGame(Turn: {self.turns_taken}, Player: {self.current_player}, Status: {status}{winner_str})"

    def __repr__(self):
        """Provides a more detailed representation."""
        return f"<BattleshipGame Turn: {self.turns_taken}, Player: {self.current_player}, Over: {self.game_over}>"


# Example Usage Block (for standalone testing)
if __name__ == "__main__":
     # Configure logging for direct script execution
     logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', force=True)

     logger.info("--- Running BattleshipGame Standalone Example ---")

     # Create game instance
     game = BattleshipGame()
     print(game)

     # Display initial boards (optional)
     print(game.render_board(0))
     print(game.render_board(1))

     # Display initial opponent views (should be all empty)
     print(game.render_opponent_view(0))
     print(game.render_opponent_view(1))

     # --- Simulate a few turns ---
     max_sim_turns = 20
     turn = 0
     while not game.game_over and turn < max_sim_turns:
         turn += 1
         current_player = game.current_player
         logger.info(f"\n--- Turn {game.turns_taken + 1}, Player {current_player} ---")

         valid_actions = game.get_valid_actions(current_player)
         if not valid_actions:
             logger.warning("No valid actions left, ending simulation.")
             break

         # Simple strategy: Choose a random valid action
         action = random.choice(valid_actions)
         logger.info(f"Player {current_player} fires at {action}.")

         result = game.fire(action)
         logger.info(f"Result: {result}")

         # Render views after the shot
         print(game.render_opponent_view(current_player)) # Show the view of the player who just shot
         if result['result'] == 'hit':
             print(f"Target Board (Player {1 - current_player}):")
             print(game.render_board(1 - current_player)) # Show opponent's board if hit


     # --- Game End ---
     if game.game_over:
         logger.info(f"--- Game Over ---")
         logger.info(f"Winner: Player {game.winner}")
         logger.info(f"Total Turns: {game.turns_taken}")
     elif turn >= max_sim_turns:
         logger.info(f"--- Simulation ended after {max_sim_turns} turns (Game not finished) ---")

     logger.info("--- BattleshipGame Standalone Example Finished ---")

# Potential Improvements noted during implementation:
# - Robustness in place_ships() against rare impossible placements.
# - Clearer handling/logging of inconsistency errors (e.g., hit registered on board but not by ship object).
# - Potentially optimize get_valid_actions if performance becomes an issue for very large boards/many turns.
# - Consider adding helper functions for common checks (e.g., is_on_board).
# - String rendering could use better characters or formatting.