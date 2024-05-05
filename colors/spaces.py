"""This module contains classes for various color spaces and methods to convert between them."""

from abc import ABC
from math import atan2, cos, degrees, radians, sin, sqrt
from typing import cast, TypeVar

from numpy import array


class ColorSpace(ABC):
    """Base class for color spaces."""

    def __init__(self, values: tuple[float, float, float]) -> None:
        self.values = values

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{self.values}"


class XYZ(ColorSpace):
    """The CIE 1931 XYZ color space."""

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

    # https://en.wikipedia.org/wiki/CIE_1931_color_space#CIE_xy_chromaticity_diagram_and_the_CIE_xyY_color_space
    @classmethod
    def from_xyY(cls, xyy: "xyY") -> "XYZ":  # noqa N801
        x, y, _Y = xyy.values
        return cls((x * _Y / y, _Y, (1 - x - y) * _Y / y))

    # https://en.wikipedia.org/wiki/CIELAB_color_space#Converting_between_CIELAB_and_CIEXYZ_coordinates
    @classmethod
    def from_CIELab(cls, cielab: "CIELab") -> "XYZ":  # noqa N801
        _L, _a, _b = cielab.values
        _X = D65.values[0] * cielab.f_inv((_L + 16) / 116 + _a / 500)
        _Y = D65.values[1] * cielab.f_inv((_L + 16) / 116)
        _Z = D65.values[2] * cielab.f_inv((_L + 16) / 116 - _b / 200)
        return cls((_X, _Y, _Z))

    # https://en.wikipedia.org/wiki/CIELUV#XYZ_%E2%86%92_CIELUV_and_CIELUV_%E2%86%92_XYZ_conversions
    @classmethod
    def from_CIELuv(cls, cieluv: "CIELuv") -> "XYZ":  # noqa N801
        _L, u, v = cieluv.values
        u_0 = cieluv.u_prime(D65)
        v_0 = cieluv.v_prime(D65)
        _Y = ((_L + 16) / 116) ** 3 if _L > 8 else _L * 27 / 24389
        a = 1 / 3 * ((52 * _L) / (u + 13 * _L * u_0) - 1)
        b = -5 * _Y
        c = -1 / 3
        d = _Y * ((39 * _L) / (v + 13 * _L * v_0) - 5)
        _X = (d - b) / (a - c)
        _Z = _X * a + b
        return cls((_X, _Y, _Z))

    # https://en.wikipedia.org/wiki/SRGB#From_sRGB_to_CIE_XYZ
    @classmethod
    def from_sRGB(cls, srgb: "sRGB") -> "XYZ":  # noqa N801
        _M = array(
            [
                [0.4124564, 0.3575761, 0.1804375],
                [0.2126729, 0.7151522, 0.0721750],
                [0.0193339, 0.1191920, 0.9503041],
            ]
        )
        srgb = array(srgb.gamma_expand().values)
        return cls(cast(tuple[float, float, float], tuple(_M @ srgb)))


D65 = XYZ((0.95047, 1.0, 1.08883))


class xyY(ColorSpace):  # noqa N801
    """
    The xyY color space is a chromaticity space that is derived from the XYZ color space.
    With Y being the luminance from the XYZ color space, x and y are the chromaticity coordinates.
    """

    @property
    def XYZ(self) -> "XYZ":  # noqa N801
        return XYZ.from_xyY(self)

    # https://en.wikipedia.org/wiki/CIE_1931_color_space#CIE_xy_chromaticity_diagram_and_the_CIE_xyY_color_space
    @classmethod
    def from_XYZ(cls, xyz: "XYZ") -> "xyY":  # noqa N801
        _X, _Y, _Z = xyz.values
        s = _X + _Y + _Z
        return cls((_X / s, _Y / s, _Y))


UCS = TypeVar("UCS", bound="UniformColorSpace")


class UniformColorSpace(ColorSpace):
    """
    Base class for uniform color spaces.
    Uniform color spaces are built such that the same geometrical distance anywhere in the color space reflects the same
    amount of perceived color difference.
    """

    @property
    def LCh(self) -> "LCh":  # noqa N801
        _L, x, y = self.values
        _C = sqrt(x**2 + y**2)
        h = degrees(atan2(y, x))
        if h < 0:
            h += 360
        return LCh((_L, _C, h))

    @classmethod
    def from_LCh(cls, lch: "LCh") -> UCS:  # noqa N801
        _L, _C, h = lch.values
        h = radians(h)
        return cls((_L, _C * cos(h), _C * sin(h)))


class CIELab(UniformColorSpace):
    """
    The CIELAB color space is a color space that is designed to be more perceptually uniform.
    L* represents the lightness of the color, a* represents the redness or greenness of the color,
    and b* represents the yellowness or blueness of the color.
    """

    @property
    def XYZ(self):  # noqa N801
        return XYZ.from_CIELab(self)

    # https://en.wikipedia.org/wiki/CIELAB_color_space#Converting_between_CIELAB_and_CIEXYZ_coordinates
    @classmethod
    def from_XYZ(cls, xyz: "XYZ") -> "CIELab":  # noqa N801
        fx, fy, fz = map(cls.f, (xyz.values[i] / D65.values[i] for i in range(3)))
        return cls(
            (
                116 * fy - 16,
                500 * (fx - fy),
                200 * (fy - fz),
            )
        )

    @staticmethod
    def f(t: float) -> float:
        if t > 216 / 24389:
            return t ** (1 / 3)
        return 841 / 108 * t + 4 / 29

    @staticmethod
    def f_inv(t: float) -> float:
        if t > 6 / 29:
            return t**3
        return 108 / 841 * (t - 4 / 29)


class CIELuv(UniformColorSpace):
    """
    The CIELUV color space is a color space that is designed to be more perceptually uniform.
    L* represents the lightness of the color, u* represents the redness or greenness of the color,
    and v* represents the yellowness or blueness of the color.
    """

    @property
    def XYZ(self):  # noqa N801
        return XYZ.from_CIELuv(self)

    # https://en.wikipedia.org/wiki/CIELUV#XYZ_%E2%86%92_CIELUV_and_CIELUV_%E2%86%92_XYZ_conversions
    @classmethod
    def from_XYZ(cls, xyz: "XYZ") -> "CIELuv":  # noqa N801
        if xyz.values[1] > 216 / 24389:
            _Y = xyz.values[1] / D65.values[1]
            _L = 116 * _Y ** (1 / 3) - 16
        else:
            _L = 24389 / 27 * xyz.values[1] / D65.values[1]
        u = 13 * _L * (cls.u_prime(xyz) - cls.u_prime(D65))
        v = 13 * _L * (cls.v_prime(xyz) - cls.v_prime(D65))
        return cls((_L, u, v))

    @staticmethod
    def u_prime(xyz: "XYZ") -> float:
        _X, _Y, _Z = xyz.values
        return 4 * _X / (_X + 15 * _Y + 3 * _Z)

    @staticmethod
    def v_prime(xyz: "XYZ") -> float:
        _X, _Y, _Z = xyz.values
        return 9 * _Y / (_X + 15 * _Y + 3 * _Z)


class LCh(ColorSpace):
    """
    The LCh color space is a cylindrical representation of a uniform color space with a lightness component.
    """

    @property
    def CIELab(self) -> "CIELab":  # noqa N801
        return CIELab.from_LCh(self)

    @property
    def CIELuv(self) -> "CIELuv":  # noqa N801
        return CIELuv.from_LCh(self)


class sRGB(ColorSpace):  # noqa N801
    """
    The sRGB color space is a standard RGB color space that is used for displays and the web.
    """

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

    def gamma_expand(self) -> "sRGB":
        def expand(c: float) -> float:
            if c <= 0.04045:
                return c / 12.92
            return ((c + 0.055) / 1.055) ** 2.4

        self.values = tuple(map(expand, self.values))
        return self

    def gamma_compress(self) -> "sRGB":
        def compress(c: float) -> float:
            if c <= 0.0031308:
                return 12.92 * c
            return 1.055 * c ** (1 / 2.4) - 0.055

        self.values = tuple(map(compress, self.values))
        return self

    # https://en.wikipedia.org/wiki/SRGB#From_CIE_XYZ_to_sRGB
    @classmethod
    def from_XYZ(cls, xyz: "XYZ") -> "sRGB":  # noqa N801
        _M = array(
            [
                [3.2404542, -1.5371385, -0.4985314],
                [-0.9692660, 1.8760108, 0.0415560],
                [0.0556434, -0.2040259, 1.0572252],
            ]
        )
        xyz = array(xyz.values)
        return cls(cast(tuple[float, float, float], tuple(_M @ xyz))).gamma_compress()

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
    """
    The HSV color space is a cylindrical representation of an RGB color space with a hue component.
    """

    @property
    def sRGB(self) -> "sRGB":  # noqa N801
        return sRGB.from_HSV(self)
