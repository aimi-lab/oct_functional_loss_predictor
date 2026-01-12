import math


class ETDRSGrid:
    def __init__(self):
        self.radii = {"center": 0.5, "inner": 1.5, "outer": 3.0}
        self.quadrants = {"superior", "temporal", "inferior", "nasal"}

    def get_area(self, ring: str) -> float:
        """Return the area of the specified ETDRS ring in mm^2.

        Args:
            ring (str): The ETDRS ring to calculate the area of. Must be one of "center", "inner", "outer", or "total".

        Returns:
            float: The area of the specified ETDRS ring in mm^2.
        """
        if ring == "center":
            return self._area(self.radii[ring])
        elif ring == "inner":
            return self._area(self.radii[ring]) - self._area(self.radii["center"])
        elif ring == "outer":
            return self._area(self.radii[ring]) - self._area(self.radii["inner"])
        elif ring == "total":
            # use summation of rings to account for floating point errors
            # area of full circle is not exactly the sum of the areas of the rings
            return (
                self.get_area("center")
                + self.get_area("inner")
                + self.get_area("outer")
            )
        else:
            raise ValueError(f"Invalid ring: {ring}")

    def get_relative_area(self, ring: str) -> float:
        """Return the relative area of the specified ETDRS ring.

        Args:
            ring (str): The ETDRS ring to calculate the relative area of. Must be one of "center", "inner", "outer", or "total".

        Returns:
            float: The relative area of the specified ETDRS ring.
        """
        return self.get_area(ring) / self.get_area("total")

    def _area(self, radius: float) -> float:
        return math.pi * radius**2
