# Generated with write_code_to_file
# /home/myuser/apps/battleshiprl/battleship/ship.py

import logging
from typing import List, Tuple

# Set up logging
# Basic configuration (level might be adjusted by the main application)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define standard Battleship ship types and their sizes
SHIPS = {
    "Carrier": 5,
    "Battleship": 4,
    "Cruiser": 3,
    "Submarine": 3,
    "Destroyer": 2,
}

class Ship:
    """
    Represents a single ship in the Battleship game.

    Attributes:
        ship_type (str): The type of the ship (e.g., "Carrier").
        size (int): The number of cells the ship occupies.
        coordinates (List[Tuple[int, int]]): List of (row, col) tuples where the ship is placed.
                                             Initialized as an empty list.
        hits (List[bool]): A boolean list tracking hits on each segment of the ship.
                           Initialized as a list of False values.
    """
    def __init__(self, ship_type: str, size: int):
        """
        Initializes a Ship instance.

        Args:
            ship_type (str): The type of the ship.
            size (int): The size (length) of the ship.
        """
        # Validate ship type and size against standard definitions, but allow non-standard ones with a warning.
        if ship_type in SHIPS and SHIPS[ship_type] != size:
             logger.warning(f"Creating ship '{ship_type}' with non-standard size {size}. Standard size is {SHIPS[ship_type]}.")
        elif ship_type not in SHIPS:
             logger.warning(f"Creating ship '{ship_type}' which is not a standard Battleship type.")

        self.ship_type: str = ship_type
        self.size: int = size
        self.coordinates: List[Tuple[int, int]] = [] # Initialize as empty
        self.hits: List[bool] = [False] * self.size # Initialize based on size
        logger.debug(f"Initialized ship: {self.ship_type} (Size: {self.size})")

    def place(self, coordinates: List[Tuple[int, int]]):
        """
        Places the ship at the specified coordinates on the board.

        Args:
            coordinates (List[Tuple[int, int]]): A list of (row, col) tuples representing
                                                 the ship's position.

        Raises:
            ValueError: If the number of coordinates does not match the ship size.
        """
        if len(coordinates) != self.size:
            logger.error(f"Attempted to place {self.ship_type} (size {self.size}) with {len(coordinates)} coordinates.")
            raise ValueError(f"Incorrect number of coordinates ({len(coordinates)}) provided for ship {self.ship_type} of size {self.size}.")

        self.coordinates = coordinates
        self.hits = [False] * self.size # Reset hits whenever the ship is placed/re-placed
        logger.debug(f"Placed ship {self.ship_type} at coordinates: {self.coordinates}")

    def is_hit(self, coordinate: Tuple[int, int]) -> bool:
        """
        Checks if the given coordinate hits this ship and updates the hit status.
        A hit is only registered (returns True) if the coordinate corresponds
        to a segment of the ship that has not been hit before.

        Args:
            coordinate (Tuple[int, int]): The (row, col) tuple to check.

        Returns:
            bool: True if the coordinate is part of the ship and was not previously hit, False otherwise.
                  Returns False if the coordinate misses the ship OR if it hits a segment
                  that was already marked as hit.
        """
        logger.debug(f"Checking if coordinate {coordinate} hits ship {self.ship_type} at {self.coordinates}")
        if coordinate in self.coordinates:
            try:
                index = self.coordinates.index(coordinate)
                if not self.hits[index]: # Check if this specific segment is already hit
                    self.hits[index] = True # Mark this segment as hit
                    # logger.info(f"HIT registered on {self.ship_type} at {coordinate} (segment index {index})")
                    return True # Return True because it's a *new* hit
                else:
                    # Coordinate is part of the ship, but this segment was already hit
                    logger.debug(f"Coordinate {coordinate} on {self.ship_type} was already hit (segment index {index}). No state change.")
                    return False # Return False as it's not a *new* hit
            except ValueError:
                 # This should technically not happen if coordinate is present in self.coordinates
                 logger.error(f"Inconsistency: Coordinate {coordinate} found in {self.ship_type}.coordinates but list.index() failed.")
                 return False # Treat as miss/error
            except IndexError:
                 # This indicates a mismatch between coordinates list and hits list size
                 logger.error(f"Index error looking for {coordinate} in {self.ship_type}. Index derived was out of bounds for hits list (size {len(self.hits)}).")
                 return False # Treat as miss/error
        else:
            # Coordinate is not part of this ship
            logger.debug(f"Coordinate {coordinate} is not part of ship {self.ship_type}")
            return False # Return False as it's a miss

    def is_sunk(self) -> bool:
        """
        Checks if all segments of the ship have been hit.

        Returns:
            bool: True if the ship is sunk (all segments are hit), False otherwise. Returns False
                  if the ship hasn't been placed yet (no coordinates).
        """
        if not self.coordinates: # Cannot be sunk if not placed
             return False
        sunk = all(self.hits)
        if sunk:
            # Log only once when it becomes sunk? If method called frequently, this might be verbose.
            # Use logger.debug or manage state to log only on transition. For now, use debug.
            logger.debug(f"Ship {self.ship_type} has been confirmed sunk (all {self.size} segments hit).")
        return sunk

    def get_coordinates(self) -> List[Tuple[int, int]]:
        """
        Returns the list of coordinates occupied by the ship.

        Returns:
            List[Tuple[int, int]]: The list of (row, col) coordinates. Returns an empty list
                                    if the ship has not been placed.
        """
        return self.coordinates

    def __str__(self) -> str:
        """
        Returns a simple string representation of the ship (its type and size).
        """
        return f"{self.ship_type} (Size: {self.size})"

    def __repr__(self) -> str:
        """
        Returns a detailed string representation of the ship, including its
        type, size, coordinates (if placed), and current hit status.
        """
        if self.coordinates:
            status = "Sunk" if self.is_sunk() else f"{sum(self.hits)}/{self.size} hits"
            coord_str = str(self.coordinates)
        else:
            status = "Not placed"
            coord_str = "N/A"
        return f"Ship(type='{self.ship_type}', size={self.size}, coords={coord_str}, status='{status}')"

# Example usage block
if __name__ == "__main__":
    # Configure logging for demonstration when run directly
    logging.basicConfig(level=logging.DEBUG, force=True) # Show debug messages for example
    logger.info("Starting Ship class example usage...")

    # Create a standard ship (Cruiser)
    cruiser = Ship("Cruiser", SHIPS["Cruiser"])
    print(f"Created ship: {cruiser}") # Uses __str__
    print(f"Details: {repr(cruiser)}") # Uses __repr__

    # Place the Cruiser
    cruiser_coords = [(2, 3), (3, 3), (4, 3)] # Vertical placement example
    try:
        cruiser.place(cruiser_coords)
        print(f"Placed {cruiser.ship_type} at {cruiser.get_coordinates()}")
        print(f"Details after placement: {repr(cruiser)}")
    except ValueError as e:
        print(f"Error placing ship: {e}")

    # Test hitting the ship
    print("\n--- Testing Hits ---")
    print(f"Is (3, 3) a hit? {cruiser.is_hit((3, 3))}") # Expected: True (new hit)
    print(f"Details: {repr(cruiser)}")
    print(f"Is (3, 3) a hit again? {cruiser.is_hit((3, 3))}") # Expected: False (already hit)
    print(f"Details: {repr(cruiser)}")
    print(f"Is (5, 5) a hit (miss)? {cruiser.is_hit((5, 5))}") # Expected: False (miss)
    print(f"Details: {repr(cruiser)}")

    # Test sinking the ship
    print("\n--- Testing Sinking ---")
    print(f"Is {cruiser.ship_type} sunk? {cruiser.is_sunk()}") # Expected: False
    print(f"Hitting remaining parts...")
    print(f"Is (2, 3) a hit? {cruiser.is_hit((2, 3))}") # Expected: True
    print(f"Is (4, 3) a hit? {cruiser.is_hit((4, 3))}") # Expected: True
    print(f"Details: {repr(cruiser)}")
    print(f"Is {cruiser.ship_type} sunk now? {cruiser.is_sunk()}") # Expected: True

    # Test creating a non-standard ship
    print("\n--- Testing Non-Standard Ship ---")
    mega_ship = Ship("MegaCarrier", 7) # Not in SHIPS dict
    print(repr(mega_ship))

    # Test placing with wrong number of coordinates
    print("\n--- Testing Invalid Placement ---")
    try:
        cruiser.place([(0,0), (0,1)]) # Trying to place size 3 ship with 2 coordinates
    except ValueError as e:
        print(f"Caught expected error: {e}")

    logger.info("Finished Ship class example usage.")

# Potential improvements/Comments from original provided code:
# - Add validation in `place` to ensure coordinates are contiguous and within board bounds (though bounds check might belong in the Game class).
# - `is_hit` logic returns True only on the *first* hit to a segment. Some game logic might need to know if a coordinate *belongs* to the ship, regardless of hit status. A separate method like `occupies(coordinate)` could be added.
# - Consider making `coordinates` and `hits` private attributes (`_coordinates`, `_hits`) with getter methods if stricter encapsulation is desired. (Current implementation uses public attributes).