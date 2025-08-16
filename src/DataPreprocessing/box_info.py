"""
This module defines the BoxInfo class
used to store information about a bounding box in a video frame.
"""

# player_ID  x1    y1   x2    y2   frame_ID  lost  grouping  generated  category
# 0          1002  436  1077  570  3586      0     1         0          digging  => line


class BoxInfo:
    """
    Class to store information about a bounding box in a video frame.
    """

    def __init__(self, line: str, debug: bool = False) -> None:
        """
        Initialize the BoxInfo object from a line of text.

        Args:
            line (str): A string containing box information in the format:
                "player_ID x1 y1 x2 y2 frame_ID lost grouping generated category"
            debug (bool): If True, prints the line being processed for debugging purposes.
        Attributes:
            line (str): The original line of text.
            debug (bool): Whether to print debug information.
            player_ID (int): The ID of the player.
            x1 (int): The x-coordinate of the top-left corner of the bounding box.
            y1 (int): The y-coordinate of the top-left corner of the bounding box.
            x2 (int): The x-coordinate of the bottom-right corner of the bounding box.
            y2 (int): The y-coordinate of the bottom-right corner of the bounding box.
            frame_ID (int): The ID of the frame.
            lost (int): Indicates if the box is lost (0 or 1).
            grouping (int): the grouping of the box (0 or 1).
            generated (int): if the box is generated (0 or 1).
            category (str): The category of the box (Player) (e.g., 'digging', 'spiking').
        """                                      
        self.line = line
        self.debug = debug
        if self.debug:
            print("f[INFO] {__class__.__name__} class Initialized...")

        info_parts  = self.line.split()


        (
            self.player_id, 
            x1, y1, x2, y2,
            self.frame_id, 
            self.lost, 
            self.grouping, 
            self.generated, 
            self.category
        ) = [
            int(part) if idx < len(info_parts) - 1 else part
            for idx, part in enumerate(info_parts)
        ]

        self.box = x1, y1, x2, y2

    def print_debug_info(self) -> None:
        """Prints detailed information about the bounding box."""

        print(f"[DEBUG] Player ID: {self.player_id}")
        print(f"[DEBUG] Box Coordinates: {self.box}")
        print(f"[DEBUG] Frame ID: {self.frame_id}")
        print(f"[DEBUG] Lost: {self.lost}, Grouping: {self.grouping}, Generated: {self.generated}")
        print(f"[DEBUG] Category: {self.category}")
    
    def __str__(self) -> str:
        """String representation of the bounding box."""

        # print(str(box))
        return (f"Player {self.player_id} | Frame {self.frame_id} | Box {self.box} | "
                f"Category: {self.category}")
    

    def __repr__(self) -> str:
        """Developer-friendly string representation."""

        # print(repr(box))
        return (f"BoxInfo(player_id={self.player_id}, box={self.box}, "
                f"frame_id={self.frame_id}, category='{self.category}')")

    def __eq__(self, other) -> bool:
        """Check if two BoxInfo objects have the same player and frame."""

        if not isinstance(other, BoxInfo):
            return NotImplemented
        return self.player_id == other.player_id and self.frame_id == other.frame_id


def test():
    line =" 0          1002  436  1077  570  3586      0     1         0          digging "
    box = BoxInfo(line, debug=True)
    print(f"Player ID: {box.player_id}")
    print(f"Box coordinates: {box.box}")
    print(f"Frame ID: {box.frame_id}")
    print(f"Category: {box.category}")
    print(f"Lost: {box.lost}")
    print(f"Grouping: {box.grouping}")
    print(f"Generated: {box.generated}")
    print(f"Line: {box.line}")
    print(f"Debug: {box.debug}")




if __name__ == "__main__":
    test()

# python -m src.DataPreprocessing.box_info
