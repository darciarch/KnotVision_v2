"""KnotVision v6.6 — customer carpet photo -> indexed Wilton production design.

Stage order: intake -> rectify -> structure -> palette -> design -> production
             -> regularize -> dither -> qc -> export

Working resolution is fixed at PX_PER_KNOT = 4 px/knot; rectified.png is exactly 4W x 4H.
"""

__version__ = "6.6.0"

PX_PER_KNOT = 4

STAGES = (
    "intake",
    "rectify",
    "structure",
    "palette",
    "design",
    "production",
    "regularize",
    "dither",
    "qc",
    "export",
)

DESIGN_CLASSES = ("classic", "geometric", "gradient", "distressed", "minimal")
DESIGN_MODES = ("quantize", "segment", "edit")
