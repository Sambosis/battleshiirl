# Generated with write_code_to_file
# /home/myuser/apps/battleshiprl/visualization/renderer.py

import pygame
import os
import sys
import logging
from typing import Tuple, Optional

# Ensure the battleship package can be found
# Adjust the path as needed if your project structure is different
# or if you run this file directly without installing the package.
try:
    # Assuming execution from the root directory or battleshiprl is in PYTHONPATH
    from battleship.game import BattleshipGame, BOARD_SIZE, BOARD_EMPTY, BOARD_SHIP, BOARD_HIT, BOARD_MISS
except ImportError:
    # Fallback for direct execution or different path configurations
    logging.warning("Could not import from battleship.game, attempting relative path adjust.")
    # Simple way to add the parent directory (battleshiprl) to the path
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)
    try:
        from battleship.game import BattleshipGame, BOARD_SIZE, BOARD_EMPTY, BOARD_SHIP, BOARD_HIT, BOARD_MISS
    except ImportError as e:
        logging.error(f"Failed to import BattleshipGame and constants: {e}")
        sys.exit("Error: BattleshipGame module not found. Make sure PYTHONPATH is set correctly.")


# --- Constants ---
CELL_SIZE = 30  # Pixels per grid cell
GRID_SIZE = BOARD_SIZE # Usually 10
BOARD_WIDTH = BOARD_HEIGHT = GRID_SIZE * CELL_SIZE
PADDING = 30  # Padding around boards and elements
INFO_HEIGHT = 60 # Space for text info at the top
WINDOW_WIDTH = PADDING * 3 + BOARD_WIDTH * 2
WINDOW_HEIGHT = PADDING * 3 + INFO_HEIGHT + BOARD_HEIGHT
CAPTION = "Battleship RL Training Visualization"

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
DARK_GRAY = (100, 100, 100)
BLUE = (0, 0, 255) # Water (background)
RED = (255, 0, 0) # Used for highlighting current player maybe?
PLAYER0_COLOR = (60, 180, 75) # Greenish
PLAYER1_COLOR = (230, 25, 75) # Reddish

# Asset paths (relative to project root /home/myuser/apps/battleshiprl)
ASSET_DIR = os.path.join(os.path.dirname(__file__), '..', 'assets', 'images')
# Explicitly join with the project root passed in context
PROJECT_ROOT = "/home/myuser/apps/battleshiprl"
ASSET_DIR = os.path.join(PROJECT_ROOT, 'assets', 'images')

HIT_IMG_PATH = os.path.join(ASSET_DIR, 'hit.png')
MISS_IMG_PATH = os.path.join(ASSET_DIR, 'miss.png')
SHIP_IMG_PATH = os.path.join(ASSET_DIR, 'ship_parts.png') # Assuming one image for all ship parts

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GameRenderer:
    """
    Handles the visualization of the Battleship game using Pygame.
    Renders two boards side-by-side, representing each player's primary grid.
    """

    def __init__(self, cell_size: int = CELL_SIZE):
        """
        Initializes Pygame, sets up the window, loads assets, and prepares fonts.

        Args:
            cell_size (int): The size of each grid cell in pixels.
        """
        logger.info("Initializing GameRenderer...")
        try:
            pygame.init()
            pygame.font.init() # Explicitly initialize font module
            logger.info("Pygame initialized successfully.")
        except pygame.error as e:
            logger.error(f"Error initializing Pygame: {e}")
            sys.exit("Pygame initialization failed.")

        self.cell_size = cell_size
        self.board_pixels = GRID_SIZE * self.cell_size

        # Recalculate dimensions based on the provided cell_size
        board_width = board_height = GRID_SIZE * self.cell_size
        window_width = PADDING * 3 + board_width * 2
        window_height = PADDING * 3 + INFO_HEIGHT + board_height

        self.screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption(CAPTION)
        logger.info(f"Pygame window created with size: {window_width}x{window_height}")

        self.clock = pygame.time.Clock() # For potential FPS limiting if needed

        # Calculate board origins
        self.board0_origin = (PADDING, PADDING * 2 + INFO_HEIGHT)
        self.board1_origin = (PADDING * 2 + board_width, PADDING * 2 + INFO_HEIGHT)

        # Load assets
        self.hit_img = None
        self.miss_img = None
        self.ship_img = None
        self.load_assets()

        # Setup fonts
        try:
            self.info_font = pygame.font.SysFont('Arial', 24)
            self.label_font = pygame.font.SysFont('Arial', 20)
            self.debug_font = pygame.font.SysFont('Arial', 16) # For cell coords if needed
        except Exception as e:
             logger.warning(f"Could not load default system font 'Arial'. Using pygame default font. Error: {e}")
             # Fallback to default font
             self.info_font = pygame.font.Font(None, 30) # Default font, size 30
             self.label_font = pygame.font.Font(None, 26) # Default font, size 26
             self.debug_font = pygame.font.Font(None, 20) # Default font, size 20

        logger.info("GameRenderer initialized.")

    def load_assets(self):
        """Loads images from the assets directory and scales them."""
        logger.info(f"Loading assets from: {ASSET_DIR}")
        try:
            # Hit image
            if not os.path.exists(HIT_IMG_PATH):
                 logger.error(f"Hit image not found at: {HIT_IMG_PATH}")
                 raise FileNotFoundError(f"Asset not found: {HIT_IMG_PATH}")
            self.hit_img = pygame.image.load(HIT_IMG_PATH).convert_alpha()
            self.hit_img = pygame.transform.scale(self.hit_img, (self.cell_size, self.cell_size))
            logger.debug("Hit image loaded and scaled.")

            # Miss image
            if not os.path.exists(MISS_IMG_PATH):
                 logger.error(f"Miss image not found at: {MISS_IMG_PATH}")
                 raise FileNotFoundError(f"Asset not found: {MISS_IMG_PATH}")
            self.miss_img = pygame.image.load(MISS_IMG_PATH).convert_alpha()
            self.miss_img = pygame.transform.scale(self.miss_img, (self.cell_size, self.cell_size))
            logger.debug("Miss image loaded and scaled.")

            # Ship image
            if not os.path.exists(SHIP_IMG_PATH):
                 logger.error(f"Ship image not found at: {SHIP_IMG_PATH}")
                 raise FileNotFoundError(f"Asset not found: {SHIP_IMG_PATH}")
            self.ship_img = pygame.image.load(SHIP_IMG_PATH).convert_alpha()
            self.ship_img = pygame.transform.scale(self.ship_img, (self.cell_size, self.cell_size))
            logger.debug("Ship image loaded and scaled.")

        except pygame.error as e:
            logger.error(f"Error loading or scaling assets: {e}")
            # Decide how critical this is - maybe proceed without images? For now, exit.
            pygame.quit()
            sys.exit("Failed to load game assets.")
        except FileNotFoundError as e:
            logger.error(str(e))
            pygame.quit()
            sys.exit("Game asset file not found.")

    def draw_grid(self, screen: pygame.Surface, player_id: int, game: BattleshipGame, origin: Tuple[int, int]):
        """
        Draws a single player's main grid, including ships, hits, and misses received.

        Args:
            screen: The Pygame surface to draw on.
            player_id: The ID (0 or 1) of the player whose grid this is.
            game: The BattleshipGame instance containing the state.
            origin: The (x, y) pixel coordinates for the top-left corner of the grid.
        """
        board_state = game.boards[player_id]
        # Shots received are based on the *other* player's opponent_view
        shots_received_view = game.opponent_views[1 - player_id]
        start_x, start_y = origin

        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                px = start_x + c * self.cell_size
                py = start_y + r * self.cell_size
                rect = pygame.Rect(px, py, self.cell_size, self.cell_size)

                # Draw cell background (water)
                pygame.draw.rect(screen, BLUE, rect)
                # Draw grid lines
                pygame.draw.rect(screen, DARK_GRAY, rect, 1)

                cell_state = board_state[r, c]
                shot_state = shots_received_view[r, c] # What the opponent shot here

                image_to_blit = None

                if cell_state == BOARD_HIT:
                    image_to_blit = self.hit_img
                elif cell_state == BOARD_SHIP:
                    # If a ship part is here, but it hasn't been marked as hit on the board, draw ship part
                    image_to_blit = self.ship_img
                elif shot_state == BOARD_MISS:
                    # If the opponent missed here (and it wasn't a ship), draw miss
                    image_to_blit = self.miss_img
                # Else: cell is BOARD_EMPTY and opponent hasn't missed here, leave as water

                if image_to_blit:
                    screen.blit(image_to_blit, rect.topleft)

                # Optional: Draw coordinates on cells for debugging
                # coord_text = self.debug_font.render(f"{r},{c}", True, WHITE)
                # screen.blit(coord_text, (px + 2, py + 2))

        # Draw border around the whole board
        pygame.draw.rect(screen, BLACK, (start_x, start_y, self.board_pixels, self.board_pixels), 2)


    def draw_info(self, screen: pygame.Surface, game: BattleshipGame):
        """
        Draws game information like current player, turn count, and game status.

        Args:
            screen: The Pygame surface to draw on.
            game: The BattleshipGame instance.
        """
        # Current Turn / Player
        turn_text = f"Turn: {game.turns_taken}"
        player_text = f"Current Player: {game.current_player}"
        player_color = PLAYER0_COLOR if game.current_player == 0 else PLAYER1_COLOR

        turn_surf = self.info_font.render(turn_text, True, BLACK)
        player_surf = self.info_font.render(player_text, True, player_color)

        screen.blit(turn_surf, (PADDING, PADDING))
        # Position player text further right
        screen.blit(player_surf, (PADDING + 200, PADDING)) # Adjust position as needed

        # Game Status
        status_text = ""
        status_color = BLACK
        if game.game_over:
            status_text = f"GAME OVER! Winner: Player {game.winner}"
            status_color = RED
        elif game.turns_taken == 0:
            status_text = "Game Starting..."
        else:
             status_text = "Game in Progress"

        status_surf = self.info_font.render(status_text, True, status_color)
        # Position status text below turn/player info or centered? Let's put it more to the right
        screen.blit(status_surf, (PADDING + 450, PADDING)) # Adjust position

        # Board Labels
        label0_surf = self.label_font.render("Player 0 Board", True, PLAYER0_COLOR)
        label1_surf = self.label_font.render("Player 1 Board", True, PLAYER1_COLOR)

        screen.blit(label0_surf, (self.board0_origin[0], self.board0_origin[1] - PADDING))
        screen.blit(label1_surf, (self.board1_origin[0], self.board1_origin[1] - PADDING))

    def render(self, game: BattleshipGame):
        """
        Draws the entire game state onto the screen.

        Args:
            game: The BattleshipGame instance to render.
        """
        # Basic event handling (allows closing the window)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                logger.info("Quit event detected. Closing renderer.")
                self.close()
                # Propagate the need to quit, maybe raise a specific exception or return False?
                # For now, just closing pygame window, caller script needs to handle termination.
                # sys.exit() # Or maybe let the caller handle exit? Avoid sys.exit here.
                return False # Indicate that rendering should stop

        # Fill background
        self.screen.fill(GRAY)

        # Draw Player 0's grid
        self.draw_grid(self.screen, 0, game, self.board0_origin)

        # Draw Player 1's grid
        self.draw_grid(self.screen, 1, game, self.board1_origin)

        # Draw game info text
        self.draw_info(self.screen, game)

        # Update the display
        pygame.display.flip()

        # Optional: Limit FPS if needed, but usually called frame-by-frame by trainer
        # self.clock.tick(30)

        return True # Indicate rendering was successful

    def close(self):
        """Shuts down Pygame."""
        logger.info("Closing Pygame and GameRenderer.")
        pygame.quit()

# Example Usage (for testing the renderer directly)
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG) # Set higher for testing
    logger.info("Starting GameRenderer example usage...")

    # Create a dummy game instance
    game = BattleshipGame() # Initializes and places ships

    # Create a renderer instance
    renderer = GameRenderer()

    # --- Simulate some game actions to test rendering updates ---
    logger.info("Simulating a few moves for renderer testing...")

    # Player 0 fires (miss)
    game.fire((5, 5)) # Assume player 0 starts, fires at 5,5
    renderer.render(game)
    pygame.time.wait(1000) # Pause for 1 sec

    # Player 1 fires (find ship location for player 0)
    p0_ship_coords = game.ships[0][0].get_coordinates() # Get coords of Player 0's first ship
    if p0_ship_coords:
        coord_to_hit = p0_ship_coords[0]
        logger.info(f"Player 1 attempting to hit Player 0 at {coord_to_hit}")
        game.fire(coord_to_hit) # Player 1 hits Player 0
        renderer.render(game)
        pygame.time.wait(1000)

        # Player 0 fires again (hit Player 1)
        p1_ship_coords = game.ships[1][0].get_coordinates() # Get coords of Player 1's first ship
        if p1_ship_coords:
            coord_to_hit_p1 = p1_ship_coords[0]
            logger.info(f"Player 0 attempting to hit Player 1 at {coord_to_hit_p1}")
            game.fire(coord_to_hit_p1)
            renderer.render(game)
            pygame.time.wait(1000)

    # --- Main loop to keep window open until closed ---
    running = True
    while running:
        # Keep rendering the current state
        if not renderer.render(game): # render() returns False if QUIT event detected
              running = False

        # Check if game is over in the dummy game, just for updating status text
        if not game.game_over:
             # Simple check: If player 0 lost all ships
             if game.check_game_over(0):
                 game.game_over = True
                 game.winner = 1
             elif game.check_game_over(1):
                 game.game_over = True
                 game.winner = 0

        # Small delay to prevent busy-waiting
        pygame.time.wait(50)


    logger.info("GameRenderer example usage finished.")