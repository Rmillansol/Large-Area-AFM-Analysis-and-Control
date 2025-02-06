"""
Author: Ruben Millan-Solsona
License: MIT

"""

import numpy as np
from typing import Literal

# Type Definition ****************************************

ChannelType = Literal[  # Types of channels used
    'Forward - Topography',
    'Backward - Topography', 
    'Forward - Deflection',
    'Backward - Deflection',
    'Forward - Analyzer 1 Amplitude',
    'Backward - Analyzer 1 Amplitude',
    'Forward - Analyzer 1 Phase',
    'Backward - Analyzer 1 Phase'
]

ExtentionType = Literal[      # AFM image file extensions
    '.xyz',
    '.gwy' 
]

# DEFINITION OF CLASSES *************************************
class clImage:
    def __init__(self, channel: ChannelType, 
                 filename = '',
                 path = '',
                 size_x: float = 1.0, size_y: float = 1.0,
                 unitxy: str = "µm", unitz: str = "nm",
                 offset_x: float = 0.0, offset_y: float = 0.0,
                 lenpxx: int = 256, lenpxy: int = 256,
                 tag: str = "", matriz: np.ndarray = None):
        
        self.channel = channel
        self.path = path
        self.filename = filename
        self.size_x = size_x
        self.size_y = size_y
        self.unitxy = unitxy
        self.unitz = unitz        
        self.tag = tag
        self.offset_x = offset_x
        self.offset_y = offset_y
        if matriz is not None:
            self.matriz = matriz 
            self.lenpxx, self.lenpxy = matriz.shape
        else:
            self.matriz = np.zeros((256, 256))  # 256 x 256 zeros default matrix
            self.lenpxx = 256
            self.lenpxy = 256
        

    def info_class(self):
        print(f"Channel: {self.channel}")
        print(f"Path: {self.path}")
        print(f"Filename: {self.filename}")
        print(f"Size X: {self.size_x}")
        print(f"Size Y: {self.size_y}")
        print(f"Unit XY: {self.unitxy}")
        print(f"Unit Z: {self.unitz}")
        print(f"Offset X: {self.offset_x}")
        print(f"Offset y: {self.offset_y}")
        print(f"len X in px: {self.lenpxx}")
        print(f"len Y in px: {self.lenpxy}")
        print(f"Tag: {self.tag}")
        print(f"Matriz:\n{self.matriz}")

if __name__ == '__main__':
    # Example of use
    objeto = clImage(channel="Backward - Topography", tag="test")

    # Show object information
    objeto.info_class()
