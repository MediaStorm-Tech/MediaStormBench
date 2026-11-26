"""Reference values of a physical X-Rite ColorChecker Classic, for photographed charts.

The synthetic charts (cameralogbench.synth.chart) are rendered FROM the sRGB 8-bit approximations, so they keep those
values as their reference. A photographed chart reflects the chart's real colours, which differ from the sRGB
approximations by up to ~0.02 in xy (cyan lies outside sRGB and is clipped there), so a photographed chart is
compared against the manufacturer's measured values instead.

Source: X-Rite ColorChecker Classic, "after November 2014" formulation, xyY under D50 (2 deg), as distributed by
colour-science (CCS_COLOURCHECKERS["ColorChecker24 - After November 2014"]). Chromaticities are adapted to D65
with the Bradford transform (the camera is white-balanced to the scene light; the evidence compares in D65).
Y is used as the relative reflectance of the neutral row.
"""
from __future__ import annotations

import numpy as np

XYY_D50: dict[str, tuple[float, float, float]] = {
    "dark_skin": (0.43748, 0.37852, 0.09832), "light_skin": (0.42225, 0.37249, 0.33620),
    "blue_sky": (0.27615, 0.29838, 0.17855), "foliage": (0.37332, 0.45117, 0.13468),
    "blue_flower": (0.30289, 0.28629, 0.22872), "bluish_green": (0.28640, 0.38972, 0.41436),
    "orange": (0.52815, 0.40533, 0.31264), "purplish_blue": (0.23239, 0.21120, 0.10911),
    "moderate_red": (0.50445, 0.32706, 0.18900), "purple": (0.33347, 0.25055, 0.06277),
    "yellow_green": (0.39889, 0.50415, 0.43318), "orange_yellow": (0.49358, 0.44428, 0.42934),
    "blue": (0.20522, 0.16867, 0.05596), "green": (0.32304, 0.51042, 0.22334),
    "red": (0.56287, 0.33556, 0.12780), "yellow": (0.46776, 0.47602, 0.59930),
    "magenta": (0.42131, 0.26692, 0.18951), "cyan": (0.20913, 0.30266, 0.18061),
    "white_9.5": (0.34891, 0.36421, 0.88069), "neutral_8": (0.34558, 0.35988, 0.58997),
    "neutral_6.5": (0.34415, 0.35919, 0.36487), "neutral_5": (0.34577, 0.35909, 0.19062),
    "neutral_3.5": (0.34265, 0.35758, 0.08817), "black_2": (0.34377, 0.35606, 0.03151),
}
_BRADFORD = np.array([[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]])


def _xyz(x: float, y: float, Y: float = 1.0) -> np.ndarray:
    return np.array([x * Y / y, Y, (1.0 - x - y) * Y / y])


_W50, _W65 = _xyz(0.34567, 0.35850), _xyz(0.31270, 0.32900)
_CAT = np.linalg.inv(_BRADFORD) @ np.diag((_BRADFORD @ _W65) / (_BRADFORD @ _W50)) @ _BRADFORD


def reference_d65() -> dict[str, tuple[float, float, float]]:
    """name -> (x, y, Y) adapted to D65."""
    out = {}
    for k, (x, y, Y) in XYY_D50.items():
        X = _CAT @ _xyz(x, y, Y)
        out[k] = (float(X[0] / X.sum()), float(X[1] / X.sum()), float(Y))
    return out
