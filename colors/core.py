from .utils import color_spaces


class Color:
    def __new__(cls, *args, **kwargs) -> "Color":
        return super().__new__(cls)

    def __init__(self, space: str, values: tuple[float, float, float]) -> None:
        """
        TODO: Add docstring

        :param space:
        :param values:
        """
        self._color = color_spaces[space](values)

    def __repr__(self) -> str:
        return f"Color({self.space}, {self.values})"

    def __str__(self) -> str:
        return str(self._color)

    @property
    def space(self) -> str:
        return self._color.__class__.__name__

    @property
    def values(self) -> tuple[float, float, float]:
        return self._color.values

    @space.setter
    def space(self, space: str) -> None:
        raise NotImplementedError  # TODO: Add conversion logic
        # This might require some helper class with class methods to convert between spaces.
        # Deciding how to handle clipping and rounding is also important.

    @values.setter
    def values(self, values: tuple[float, float, float]) -> None:
        self._color.values = values
