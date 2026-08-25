"""Preliminary North Sea polygon mask for the EERIE North Sea prototype.

This is a hand- digitised, conservative polygon intended to cover the open North Sea,
the Dogger Bank area, the German Bight, the southern Baltic reference site
(Kriegers Flak), and the northern North Sea up to the latitude of the Shetlands.

It deliberately excludes the British and European mainland and the open Atlantic
west of Scotland. For a production/bankable assessment it should be replaced by an
official North Sea EEZ, OSPAR, or national hydrographic polygon.
"""

import numpy as np
import pandas as pd
from matplotlib.path import Path

NORTH_SEA_POLYGON = np.array(
    [
        (-5.0, 50.0),
        (2.0, 50.0),
        (5.0, 51.0),
        (8.0, 52.0),
        (10.0, 53.5),
        (13.0, 54.5),
        (13.0, 56.0),
        (11.0, 57.5),
        (8.0, 58.5),
        (4.0, 59.5),
        (0.0, 59.0),
        (-3.0, 58.5),
        (-5.0, 56.5),
        (-5.0, 50.0),
    ]
)

_MASK_PATH = Path(NORTH_SEA_POLYGON)


def mask_for_points(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Return a boolean mask that is True for points inside the North Sea polygon."""
    points = np.column_stack([lon, lat])
    return _MASK_PATH.contains_points(points)


def apply_mask(frame: pd.DataFrame, lon_col: str = "lon", lat_col: str = "lat") -> pd.DataFrame:
    """Return the subset of `frame` that falls inside the North Sea polygon."""
    mask = mask_for_points(frame[lon_col].to_numpy(), frame[lat_col].to_numpy())
    return frame.loc[mask].copy()
