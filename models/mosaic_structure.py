from dataclasses import dataclass, field

import cv2
import numpy as np

"""
Mosaic pattern
╔═══════════════╦═══════════════╦═...
║(x1,y1)        ║(x1,y1)        ║
║     Tile 0    ║     Tile 1    ║
║        (x2,y2)║        (x2,y2)║
╠═══════════════╬═══════════════╬═...
║(x1,y1)        ║(x1,y1)        ║
║     Tile 2    ║     Tile 3    ║
║        (x2,y2)║        (x2,y2)║
╠═══════════════╬═══════════════╬═...
"""

# Text variables
FONT = cv2.FONT_HERSHEY_SIMPLEX
LINE_TYPE = cv2.LINE_AA
SIZE = 0.6
COLOR = (0, 0, 255)
PAD = 10  # Pixels


@dataclass(slots=True)
class MosaicStructure:
    """
    Data class for storing and modifying mosaic content.
    """
    tile_number: int
    tile_grid: list[int]  # (rows, columns)
    tile_resolution: list[int]  # (height, width)
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
        self.mosaic_content = np.zeros(
            shape=(self.tile_grid[0] * self.tile_resolution[0],
                   self.tile_grid[1] * self.tile_resolution[1], 3),
            dtype=np.uint8
        )  # blank mosaic

    def __repr__(self):
        return (
            f"Mosaic Structure ---------------------\n"
            f"- Tile grid:\t\t{self.tile_grid[0]} Rows × {self.tile_grid[1]} Columns\n"
            f"- Tile resolution:\t{self.tile_resolution[0]}×{self.tile_resolution[1]}"
        )

    def insert_tile_content(self, index: int, content: np.ndarray, text=""):
        y1, y2, x1, x2 = self.tile_positions[index]
        self.mosaic_content[y1:y2, x1:x2] = content
        cv2.putText(
            self.mosaic_content, text, (x1 + PAD, y2 - PAD),
            FONT, SIZE, COLOR, 1, LINE_TYPE
        )


if __name__ == "__main__":
    pass
