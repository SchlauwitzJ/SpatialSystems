from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Union


SPATIAL_KEYS = ("+0", "-0", "+1", "-1", "+2", "-2", "+3", "-3")
COM_ROT = {"++": 1, "+-": 1, "-+": 1, "--": -1}
GEO_ROT = {"12": 1, "21": -1, "13": 1, "31": -1, "23": 1, "32": -1,
           "01": 1, "02": 1, "03": 1, "10": 1, "20": 1, "30": 1,
           "00": 1, "11": 1, "22": 1, "33": 1}

GEO_SHAPE = {"scalar": ("+0",),
             "vector": ("+1", "+2", "+3"),
             "bi-vector": ("-1", "-2", "-3"),
             "tri-vector": ("-0",)}
GEO_SHAPE2 = {"scalars": ("+0", "-0"),
              "vectors": ("+1", "+2", "+3", "-1", "-2", "-3")}

GEO_SPATIAL_KEYS = {
    "+0": {"+0": "+0", "-0": "-0", "+1": "+1", "-1": "-1", "+2": "+2", "-2": "-2", "+3": "+3", "-3": "-3"},
    "-0": {"+0": "-0", "-0": "+0", "+1": "-1", "-1": "+1", "+2": "-2", "-2": "+2", "+3": "-3", "-3": "+3"},
    "+1": {"+0": "+1", "-0": "-1", "+1": "+0", "-1": "-0", "+2": "-3", "-2": "+3", "+3": "-2", "-3": "+2"},
    "-1": {"+0": "-1", "-0": "+1", "+1": "-0", "-1": "+0", "+2": "+3", "-2": "-3", "+3": "+2", "-3": "-2"},
    "+2": {"+0": "+2", "-0": "-2", "+1": "-3", "-1": "+3", "+2": "+0", "-2": "-0", "+3": "-1", "-3": "+1"},
    "-2": {"+0": "-2", "-0": "+2", "+1": "+3", "-1": "-3", "+2": "-0", "-2": "+0", "+3": "+1", "-3": "-1"},
    "+3": {"+0": "+3", "-0": "-3", "+1": "-2", "-1": "+2", "+2": "-1", "-2": "+1", "+3": "+0", "-3": "-0"},
    "-3": {"+0": "-3", "-0": "+3", "+1": "+2", "-1": "-2", "+2": "+1", "-2": "-1", "+3": "-0", "-3": "+0"}}


@dataclass
class GeoSpatial:
    """
    For vectorizing spatially oriented activities.
    The functions assume the dictionary input is compliant with the result of get_shape().
    z = x & x => dot/inner
    z = x | x => outer
    z = x * x => geometric
    z = x / x => 1/x geometric

    !!!!!!!!!!!!!!!!!! USE BRACKETS TO ENSURE PROPER ORDER OF OPERATION !!!!!!!!!!!!!!!!!!!!!
    """

    def __init__(self, src: Union[dict, GeoSpatial] = None):
        """
        dimension 0 is assumed to be scalar while others are part of a vector/bi-/tri-.
        The layout is {+0, +1, +2, +3, -1, -2, -3, -0}
        """

        if src is None:
            self.__vec = {sp_key: 0.0 for sp_key in SPATIAL_KEYS}
        else:
            self.__vec = {sp_key: src.get(sp_key, 0.0) for sp_key in SPATIAL_KEYS}

    def keys(self):
        return self.__vec.keys()

    def zero(self, dim: Union[str, list, tuple, set] = None) -> None:
        """
        Set all specified elements of this Geo to zero.
        Clear all dimensions if none are specified.

        :param dim:
        :return:
        """
        if dim is None:
            for dim in self.keys():
                self[dim] = 0.0
        elif isinstance(dim, (list, tuple, set)):
            for key in dim:
                self[key] = 0.0
        else:
            for ky in GEO_SHAPE[dim]:
                self[ky] = 0.0
        return

    def scale(self, value=1.0, dim: Union[str, list, tuple, set] = None) -> GeoSpatial:
        """
        Get a scaled version of this Geo.
        :param value:
        :param dim:
        :return:
        """
        n_self = GeoSpatial(src=self)

        if dim is None:
            for key in n_self.keys():
                n_self[key] *= value
        elif isinstance(dim, (list, tuple, set)):
            for key in dim:
                n_self[key] *= value
        else:
            for ky in GEO_SHAPE[dim]:
                n_self[ky] *= value
        return n_self

    def subset(self, typ="scalar") -> GeoSpatial:
        """
        Get the sub-set Geo based on the global GEO_SHAPE.
        :param typ:
        :return:
        """
        if typ in GEO_SHAPE.keys():
            return GeoSpatial(src={ky: self[ky] for ky in GEO_SHAPE[typ]})
        raise KeyError(f"{typ} is not a valid key for geometrics.")

    def conj(self) -> GeoSpatial:
        """
        The complex conjugate of the Geo.
        :return:
        """
        nw_spc = GeoSpatial()
        for ky, val in self:
            if ky[0] == "+":
                nw_spc[ky] = val
            else:
                nw_spc[ky] = -val
        return nw_spc

    def scale_sq(self) -> float:
        """
        Get the scale squared of this Geo. I.e. ||x||**2
        :return:
        """
        return (self & self.conj())[SPATIAL_KEYS[0]]

    def one_over(self) -> GeoSpatial:
        """
        Get 1/x of this Geo.
        :return:
        """
        nw_spc = self.conj()
        nw_spc /= self.scale_sq()
        return nw_spc

    def get(self, key, default) -> float:
        if key not in self.keys():
            return default
        return self[key]

    def __iter__(self):
        return iter([(ky, self[ky]) for ky in self.keys()])

    def __getitem__(self, key: str) -> float:
        """
        it is assumed keys are from the SPATIAL_KEYS tuple.
        :param key:
        :return:
        """
        if key in self.keys():
            return self.__vec[key]
        raise KeyError(f"{key} does not exist in {self.keys()}.")

    def __setitem__(self, key: str, value: float) -> None:
        """

        :param key:
        :param value:
        :return:
        """
        if key in self.keys():
            self.__vec[key] = value
        else:
            raise KeyError(f"{key} does not exist in {self.keys()}.")
        return

    def __add__(self, other: Union[dict, float: 0.0, int, bool, GeoSpatial]) -> GeoSpatial:
        """
        other can also be a sub-set of the full space.
        :param other:
        :return:
        """
        nw_spc = GeoSpatial()
        if isinstance(other, (GeoSpatial, dict)):
            for key in other.keys():
                nw_spc[key] = self[key] + other[key]
        elif isinstance(other, (float, int, bool)):
            nw_spc[SPATIAL_KEYS[0]] += other
        else:
            raise TypeError(f"{type(other)} is not a valid type for addition.")
        return nw_spc

    def __sub__(self, other: Union[dict, float: 0.0, int, bool, GeoSpatial]) -> GeoSpatial:
        """
        other can also be a sub-set of the full space.
        :param other:
        :return:
        """
        nw_spc = GeoSpatial()
        if isinstance(other, (GeoSpatial, dict)):
            for key in other.keys():
                nw_spc[key] = self[key] - other[key]
        elif isinstance(other, (float, int, bool)):
            nw_spc[SPATIAL_KEYS[0]] -= other
        else:
            raise TypeError(f"{type(other)} is not a valid type for subtraction.")
        return nw_spc

    def __neg__(self):

        nw_spc = GeoSpatial()
        for key in self.keys():
            nw_spc[key] = -self[key]
        return nw_spc

    def __and__(self, other: Union[dict, GeoSpatial]) -> GeoSpatial:
        """
        other can also be a sub-set of the full space.
        This is equivalent to a vector inner(dot)-product
        :param other: contains a vector and/or bi-vector
        :return: a geospatial set with the resulting real scalar in +0
        """
        nw_spc = GeoSpatial()
        for o_key in other.keys():
            nw_spc[GEO_SPATIAL_KEYS[o_key][o_key]] += self[o_key] * other[o_key] * COM_ROT[o_key[0] + o_key[0]]

        return nw_spc

    def __xor__(self, other: Union[dict, GeoSpatial]) -> GeoSpatial:
        """
        other can also be a sub-set of the full space.
        This is equivalent to a vector outer-product
        :param other: contains a vector and/or bi-vector
        :return: a geospatial set with the resulting vector in (+1, +2, +3)
        """
        nw_spc = GeoSpatial()
        for o_key in other.keys():
            for s_key in self.keys():
                if s_key != o_key:
                    nw_spc[GEO_SPATIAL_KEYS[s_key][o_key]] += self[s_key] * other[o_key] * GEO_ROT[
                        o_key[1] + s_key[1]] * COM_ROT[o_key[0] + s_key[0]] * 0.5

        return nw_spc

    def __mul__(self, other: Union[dict, float: 1.0, int, bool, GeoSpatial]) -> GeoSpatial:
        """
        other can also be a sub-set of the full space.
        This is equivalent to a vector geometric-product
        :param other: contains a scalar, vector, bi-vector, and/or tri-vector
        :return: a geospatial set
        """

        if isinstance(other, (float, int, bool)):
            nw_spc = GeoSpatial(src=self).scale(value=other)
            return nw_spc

        nw_spc = GeoSpatial()
        for o_key in other.keys():
            for s_key in self.keys():
                cmpnd = self[s_key] * other[o_key] * GEO_ROT[o_key[1] + s_key[1]] * COM_ROT[o_key[0] + s_key[0]]
                if o_key != s_key:
                    cmpnd *= 0.5

                nw_spc[GEO_SPATIAL_KEYS[s_key][o_key]] += cmpnd

        return nw_spc

    def __pow__(self, power, modulo=None):
        """
        scale the elements based on the power. This is useful for euclidean operations.
        :param power:
        :param modulo:
        :return: this geometric product scaled by a power
        """
        reslt = GeoSpatial()
        for ky in self.keys():
            reslt[ky] = np.sign(self[ky]) * np.abs(self[ky]) ** power

        return reslt

    def __truediv__(self, other: Union[dict, float: 1.0, int, bool, GeoSpatial]) -> GeoSpatial:
        """
        other can also be a sub-set of the full space.
        This is equivalent to a vector geometric-product
        :param other:
        :return:
        """

        if isinstance(other, (float, int, bool)):
            nw_spc = GeoSpatial(src=self)
            return nw_spc.scale(value=1 / other)

        return self.__mul__(other=other.one_over())

    def __abs__(self) -> GeoSpatial:
        """
        rectify the elements such that they are all positive.
        :return:
        """
        n_self = GeoSpatial(src=self)

        for key in n_self.keys():
            n_self[key] = np.abs(n_self[key])
        return n_self

    def __dict__(self) -> dict:
        return {ky: val for ky, val in self}

    def __str__(self) -> str:
        n_str = "("
        for grp in GEO_SHAPE.values():
            for dim in grp:
                n_str += f"{dim}: {self[dim]}, "
            n_str = n_str[:-1] + "), ("

        n_str = n_str[:-3]

        return n_str


@dataclass
class GeoMatrix:
    def __init__(self):
        pass
