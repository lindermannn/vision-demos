"""Chilean mining safety rule for berm height, as a single source of truth.

DS 132, Reglamento de Seguridad Minera, states the requirement as a fraction of
the *wheel* of the vehicle that operates there:

* **Artículo 342, letra b** (waste dumps): "Diseño y construcción de bermas de
  protección efectivas en los bordes. El cordón de seguridad en el borde deberá
  tener una altura mínima de 1/2 rueda del camión de mayor envergadura que
  descargue en él."
* **Artículo 351** (steep roads with curves or running above ravines): "se debe
  disponer de un pretil, a la orilla exterior del camino, con una altura mínima
  de 2/3 de la altura de la rueda del equipo o vehículo que circulará por el
  lugar."

The corpus of this assessment is a waste dump, so Artículo 342 b is the rule in
force and Artículo 351 is reported alongside it as the stricter road case.

Because the requirement is a fraction of a wheel, and a wheel has a known
diameter for a given truck class, the threshold resolves to a fixed number of
metres once that class is declared.  Metres are therefore the primary unit here
and the wheel ratio is kept as the secondary, scale-free reading.

Declared machine assumptions, and the only ones in this module:

* Tyre 40.00R57, 3.596 m inflated outside diameter, the size fitted to
  793-class rigid haul trucks.
* Cat 793F transport height 6.6 m.

Both are stated in the report; a different fleet changes the numbers and only
these constants have to move.
"""

from __future__ import annotations

from dataclasses import dataclass


WHEEL_DIAMETER_M = 3.596
TRUCK_HEIGHT_M = 6.6
WHEEL_TO_TRUCK_HEIGHT = WHEEL_DIAMETER_M / TRUCK_HEIGHT_M

DUMP_EDGE_FRACTION = 0.50          # Artículo 342 b
STEEP_ROAD_FRACTION = 2.0 / 3.0    # Artículo 351

DUMP_EDGE_MINIMUM_M = DUMP_EDGE_FRACTION * WHEEL_DIAMETER_M      # 1.80 m
STEEP_ROAD_MINIMUM_M = STEEP_ROAD_FRACTION * WHEEL_DIAMETER_M    # 2.40 m

ARTICLE_DUMP = "DS 132 Art. 342 b"
ARTICLE_ROAD = "DS 132 Art. 351"


@dataclass(frozen=True)
class Compliance:
    """Verdict for one berm height, in metres first and wheels second."""

    height_m: float
    height_m_low: float
    height_m_high: float
    required_m: float
    article: str
    verdict: str
    wheel_ratio: float

    @property
    def margin_m(self) -> float:
        return self.height_m - self.required_m


def evaluate(
    height_m: float,
    height_m_low: float,
    height_m_high: float,
    article: str = ARTICLE_DUMP,
) -> Compliance:
    """Compare a measured height against the article in force.

    The verdict is deliberately three-valued.  A point estimate that clears the
    threshold while its lower bound does not is not a compliant berm, it is an
    inconclusive measurement, and saying so is the honest reading of an estimate
    that carries a declared interval.
    """

    required = DUMP_EDGE_MINIMUM_M if article == ARTICLE_DUMP else STEEP_ROAD_MINIMUM_M
    if height_m_low >= required:
        verdict = "cumple"
    elif height_m_high < required:
        verdict = "por debajo"
    else:
        verdict = "no concluyente"
    return Compliance(
        height_m=height_m,
        height_m_low=height_m_low,
        height_m_high=height_m_high,
        required_m=required,
        article=article,
        verdict=verdict,
        wheel_ratio=height_m / WHEEL_DIAMETER_M,
    )


def summary_line(compliance: Compliance) -> str:
    """One line for the video panel: metres first, wheels second."""

    return (
        f"h={compliance.height_m:.2f} m "
        f"[{compliance.height_m_low:.2f}-{compliance.height_m_high:.2f}] | "
        f"min {compliance.required_m:.2f} m {compliance.article} | "
        f"{compliance.verdict} | {compliance.wheel_ratio:.2f} rueda"
    )
