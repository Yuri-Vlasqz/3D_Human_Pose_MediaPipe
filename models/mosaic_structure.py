import numpy as np
import cv2

from timeit import repeat
from dataclasses import dataclass, field

"""
Mosaic pattern
╔═══════════════╦═══════════════╗...
║(x1,y1)        ║(x1,y1)        ║
║     Tile 0    ║     Tile 1    ║
║        (x2,y2)║        (x2,y2)║
╠═══════════════╬═══════════════╣...
║(x1,y1)        ║(x1,y1)        ║
║     Tile 2    ║     Tile 3    ║
║        (x2,y2)║        (x2,y2)║
╚═══════════════╩═══════════════╝...
"""

# Text variables
FONT = cv2.FONT_HERSHEY_SIMPLEX
LINE_TYPE = cv2.LINE_AA
SIZE = 0.6
COLOR = (255, 255, 255)
PAD = 10  # Pixels


@dataclass(slots=True)
class MosaicStructure:
    tile_number: int
    tile_grid: list[int, int]  # (rows, columns)
    tile_resolution: list[int, int]  # (height, width)
    tile_positions: list[list] = field(init=False)
    mosaic_content: np.ndarray = field(init=False)

    def __post_init__(self):
        tile_positions = []
        for position in range(self.tile_number):
            row, col = divmod(position, self.tile_grid[1])
            y1 = row * self.tile_resolution[0]
            y2 = y1 + self.tile_resolution[0]
            x1 = col * self.tile_resolution[1]
            x2 = x1 + self.tile_resolution[1]
            tile_positions.append([y1, y2, x1, x2])

        self.tile_positions = tile_positions
        self.mosaic_content = np.zeros((
            self.tile_grid[0] * self.tile_resolution[0],
            self.tile_grid[1] * self.tile_resolution[1],
            3), dtype=np.uint8)  # empty mosaic

    def __repr__(self):
        return (f"Mosaic Structure ---------------------\n"
                f"- Tile grid:\t\t{self.tile_grid[0]} Rows × {self.tile_grid[1]} Columns\n"
                f"- Tile resolution:\t{self.tile_resolution[0]}×{self.tile_resolution[1]}")

    def insert_tile_content(self, index: int, content: np.ndarray):
        y1, y2, x1, x2 = self.tile_positions[index]
        # cv2.putText(content, text, (x1+PAD, y2-PAD), FONT, SIZE, COLOR, 1, LINE_TYPE)
        self.mosaic_content[y1:y2, x1:x2] = content


if __name__ == "__main__":
    test = MosaicStructure(6, [2, 3], [480, 480])
    test_content = np.empty((480, 480, 3), dtype=np.uint8)
    test.insert_tile_content(0, test_content)
    test.insert_tile_content(2, test_content)


    def example_function():
        global test
        for _ in range(1000):
            _ = test.mosaic_content
            
    execution_time = repeat(example_function, repeat=20, number=1)
    print(f"Execution time:\n"
          f" - Mean: {round(np.mean(execution_time)*1000, 3)} µs\n"
          f" - Max: {round(np.max(execution_time)*1000, 3)} µs\n"
          f" - Min: {round(np.min(execution_time)*1000, 3)} µs\n")
