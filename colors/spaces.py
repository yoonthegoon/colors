from abc import ABC, abstractmethod
from math import atan2, cos, degrees, radians, sin, sqrt
from typing import TypeVar


class ColorSpace(ABC):
    def __init__(self, values: tuple[float, float, float]) -> None:
        self.values = values

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{self.values}"


class XYZ(ColorSpace):
    @property
    def xyY(self) -> "xyY":  # noqa N801
        return xyY.from_XYZ(self)

    @property
    def CIELab(self) -> "CIELab":  # noqa N801
        return CIELab.from_XYZ(self)

    @property
    def CIELuv(self) -> "CIELuv":  # noqa N801
        return CIELuv.from_XYZ(self)

    @property
    def sRGB(self) -> "sRGB":  # noqa N801
        return sRGB.from_XYZ(self)

    @classmethod
    def from_xyY(cls, xyy: "xyY") -> "XYZ":  # noqa N801
        x, y, _Y = xyy.values
        return cls((x * _Y / y, _Y, (1 - x - y) * _Y / y))

    @classmethod
    def from_CIELab(cls, cielab: "CIELab") -> "XYZ":  # noqa N801
        raise NotImplementedError

    @classmethod
    def from_CIELuv(cls, cieluv: "CIELuv") -> "XYZ":  # noqa N801
        raise NotImplementedError

    @classmethod
    def from_sRGB(cls, srgb: "sRGB") -> "XYZ":  # noqa N801
        raise NotImplementedError


class xyY(ColorSpace):  # noqa N801
    @property
    def XYZ(self) -> "XYZ":  # noqa N801
        return XYZ.from_xyY(self)

    @classmethod
    def from_XYZ(cls, xyz: "XYZ") -> "xyY":  # noqa N801
        _X, _Y, _Z = xyz.values
        s = _X + _Y + _Z
        return cls((_X / s, _Y / s, _Y))


UCS = TypeVar("UCS", bound="UniformColorSpace")


class UniformColorSpace(ColorSpace):
    @property
    def LCh(self) -> "LCh":  # noqa N801
        _L, x, y = self.values
        _C = sqrt(x**2 + y**2)
        h = degrees(atan2(y, x))
        if h < 0:
            h += 360
        return LCh((_L, _C, h))

    @classmethod
    @abstractmethod
    def from_LCh(cls, lch: "LCh") -> UCS:  # noqa N801
        _L, _C, h = lch.values
        h = radians(h)
        return cls((_L, _C * cos(h), _C * sin(h)))


class CIELab(UniformColorSpace):
    @property
    def XYZ(self):  # noqa N801
        return XYZ.from_CIELab(self)

    @classmethod
    def from_XYZ(cls, xyz: "XYZ") -> "CIELab":  # noqa N801
        raise NotImplementedError

    @classmethod
    def from_LCh(cls, lch: "LCh") -> "CIELab":
        return cls(super().from_LCh(lch).values)


class CIELuv(UniformColorSpace):
    @property
    def XYZ(self):  # noqa N801
        return XYZ.from_CIELuv(self)

    @classmethod
    def from_XYZ(cls, xyz: "XYZ") -> "CIELuv":  # noqa N801
        raise NotImplementedError

    @classmethod
    def from_LCh(cls, lch: "LCh") -> "CIELuv":
        return cls(super().from_LCh(lch).values)


class LCh(ColorSpace):
    @property
    def CIELab(self) -> "CIELab":  # noqa N801
        return CIELab.from_LCh(self)

    @property
    def CIELuv(self) -> "CIELuv":  # noqa N801
        return CIELuv.from_LCh(self)


class sRGB(ColorSpace):  # noqa N801
    @property
    def XYZ(self) -> "XYZ":  # noqa N801
        return XYZ.from_sRGB(self)

    @property
    def HSV(self):  # noqa N801
        _R, _G, _B = self.values
        _M = max(_R, _G, _B)
        m = min(_R, _G, _B)
        _C = _M - m
        if _C == 0:
            _H = 0
        elif _M == _R:
            _H = 60 * ((_G - _B) / _C % 6)
        elif _M == _G:
            _H = 60 * ((_B - _R) / _C + 2)
        else:
            _H = 60 * ((_R - _G) / _C + 4)
        _V = _M
        _S = 0 if _M == 0 else _C / _M
        return HSV((_H, _S, _V))

    @classmethod
    def from_XYZ(cls, xyz: "XYZ") -> "sRGB":  # noqa N801
        raise NotImplementedError

    @classmethod
    def from_HSV(cls, hsv: "HSV") -> "sRGB":  # noqa N801
        _H, _S, _V = hsv.values
        _C = _V * _S
        _H /= 60
        _X = _C * (1 - abs(_H % 2 - 1))
        m = _V - _C
        if _H < 1:
            _R, _G, _B = _C, _X, 0
        elif _H < 2:
            _R, _G, _B = _X, _C, 0
        elif _H < 3:
            _R, _G, _B = 0, _C, _X
        elif _H < 4:
            _R, _G, _B = 0, _X, _C
        elif _H < 5:
            _R, _G, _B = _X, 0, _C
        else:
            _R, _G, _B = _C, 0, _X
        return cls((_R + m, _G + m, _B + m))


class HSV(ColorSpace):
    @property
    def sRGB(self) -> "sRGB":  # noqa N801
        return sRGB.from_HSV(self)
