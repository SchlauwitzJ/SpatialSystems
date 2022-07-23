from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Union
import cmath as cm
import json

SPATIAL_KEYS = ("+0", "-0", "+1", "-1", "+2", "-2", "+3", "-3")
RED_GEO_SHAPE = ('0', '1', '2', '3')
RED_GEO_SHAPE_UPPER = {'0': ('1', '2', '3'), '1': ('2', '3'), '2': ('3',)}

GEO_SHAPE = {"scalar": ("+0",),
             "vector": ("+1", "+2", "+3"),
             "bi-vector": ("-1", "-2", "-3"),
             "tri-vector": ("-0",),
             "scalars": ("+0", "-0"),
             "vectors": ("+1", "+2", "+3", "-1", "-2", "-3")}

# X|Y ==> GRK[X_ky][Y_ky] = (dim_red*complex_rot*(geo-product mid op))
GEO_RED_KYS = {
    "+0": {"+0": ("+0", 1),
           "-0": ("-0", 1),
           "+1": ("+1", 1),
           "-1": ("-1", 1),
           "+2": ("+2", 1),
           "-2": ("-2", 1),
           "+3": ("+3", 1),
           "-3": ("-3", 1)},
    "-0": {"+0": ("-0", 1),
           "-0": ("+0", -1),
           "+1": ("-1", 1),
           "-1": ("+1", -1),
           "+2": ("-2", 1),
           "-2": ("+2", -1),
           "+3": ("-3", 1),
           "-3": ("+3", -1)},
    "+1": {"+0": ("+1", 1),
           "-0": ("-1", 1),
           "+1": ("+0", 1),
           "-1": ("-0", 1),
           "+2": ("-3", 1),
           "-2": ("+3", 1),
           "+3": ("-2", 1),
           "-3": ("+2", 1)},
    "-1": {"+0": ("-1", 1),
           "-0": ("+1", -1),
           "+1": ("-0", 1),
           "-1": ("+0", -1),
           "+2": ("+3", 1),
           "-2": ("-3", -1),
           "+3": ("+2", 1),
           "-3": ("-2", -1)},
    "+2": {"+0": ("+2", 1),
           "-0": ("-2", 1),
           "+1": ("-3", 1),
           "-1": ("+3", 1),
           "+2": ("+0", 1),
           "-2": ("-0", 1),
           "+3": ("-1", 1),
           "-3": ("+1", 1)},
    "-2": {"+0": ("-2", 1),
           "-0": ("+2", -1),
           "+1": ("+3", 1),
           "-1": ("-3", -1),
           "+2": ("-0", 1),
           "-2": ("+0", -1),
           "+3": ("+1", 1),
           "-3": ("-1", -1)},
    "+3": {"+0": ("+3", 1),
           "-0": ("-3", 1),
           "+1": ("-2", 1),
           "-1": ("+2", 1),
           "+2": ("-1", 1),
           "-2": ("+1", 1),
           "+3": ("+0", 1),
           "-3": ("-0", 1)},
    "-3": {"+0": ("-3", 1),
           "-0": ("+3", -1),
           "+1": ("+2", 1),
           "-1": ("-2", -1),
           "+2": ("+1", 1),
           "-2": ("-1", -1),
           "+3": ("-0", 1),
           "-3": ("+0", -1)}}


@dataclass
class Geo:
    """
    For vectorizing spatially oriented activities.
    The functions assume the dictionary input is compliant with the result of get_shape().
    Cross-dimensional operators
    z = x & y => dot/inner-product
    z = x | y => outer-product
    z = x ^ y => geometric-product
    z = x.inverse() => 1/x based on vector math
    z = x @ y => rotation of x by y

    dimension-wise operators:
    z = x * y => multiplication
    z = x / y => division
    z = x + y => addition
    z = x - y => subtraction

    !!!!!!!!!!!!!!!!!! USE BRACKETS TO ENSURE PROPER ORDER OF OPERATION !!!!!!!!!!!!!!!!!!!!!
    """

    def __init__(self, src: Union[float, int, bool, complex, dict, Geo] = None, def_val=0.0):
        """
        dimension 0 is assumed to be scalar while others are part of a vector/bi-/tri-.
        The layout is {+0, +1, +2, +3, -1, -2, -3, -0}
        """

        if isinstance(src, (dict, Geo)):
            self.__vec = {sp_key: src.get(sp_key, def_val) for sp_key in SPATIAL_KEYS}
        else:
            self.__vec = {sp_key: def_val for sp_key in SPATIAL_KEYS}
            if isinstance(src, (float, int, bool)):
                self.__vec[GEO_SHAPE["scalar"][0]] = src
            elif isinstance(src, (complex,)):
                self.__vec[GEO_SHAPE["scalar"][0]] = src.real
                self.__vec[GEO_SHAPE["tri-vector"][0]] = src.imag

    def sum(self) -> float:
        tot_v = 0.0
        for val in self.values():
            tot_v += val
        return tot_v

    def magnitude_sq(self) -> float:
        """
        Get the magnitude squared of this Geo. I.e. ||x||**2
        :return:
        """
        return (self & self.conj())['+0']

    def magnitude(self) -> float:
        """
        Get the magnitude of this Geo. I.e. ||x||
        :return:
        """
        return np.sqrt(self.magnitude_sq())

    def __hash__(self) -> hash:
        return hash(tuple(self.values()))

    def __dict__(self) -> dict:
        return {ky: val for ky, val in self}

    def __str__(self) -> str:
        n_str = "Geo<"
        for dim in self.keys():
            n_str += f"{dim}: {self[dim]}, "

        n_str = n_str[:-2] + ">"

        return n_str

    def __bool__(self) -> bool:
        """
        evaluate if this set is not zero.
        :return:
        """
        for val in self.values():
            if val:
                return True
        return False

    def __reduce_ex__(self, protocol):
        return self.__class__, (self.__vec, 0.0)

    def __repr__(self) -> str:
        return json.dumps(self.to_json(), sort_keys=True, ensure_ascii=False, indent=4)

    def to_json(self):
        return self.__dict__()

    # ----- dictionary/vector-like elements
    def keys(self):
        return self.__vec.keys()

    def values(self):
        return self.__vec.values()

    def items(self):
        return self.__vec.items()

    def __index__(self) -> int:
        return len(self.keys())

    def __iter__(self) -> iter:
        return iter([(ky, self[ky]) for ky in self.keys()])

    def __getitem__(self, key: str) -> Union[float, complex, Geo]:
        """
        it is assumed keys are from the SPATIAL_KEYS tuple.
        :param key:
        :return:
        """
        if key in self.keys():
            return self.__vec[key]
        elif key in RED_GEO_SHAPE:
            return self.__vec[f"+{key}"] + self.__vec[f"-{key}"]*1j
        elif key == '+':
            return self.real()
        elif key == '-':
            return self.imag()
        if key in GEO_SHAPE.keys():
            return Geo(src={ky: self[ky] for ky in GEO_SHAPE[key]})
        raise KeyError(f"{key} does not exist in {self.keys()}.")

    def get(self, key, default) -> Union[float, complex, Geo]:
        if key in self.keys():
            return self.__vec[key]
        elif key in RED_GEO_SHAPE:
            return self.__vec[f"+{key}"] + self.__vec[f"-{key}"]*1j
        elif key == '+':
            return self.real()
        elif key == '-':
            return self.imag()
        if key in GEO_SHAPE.keys():
            return Geo(src={ky: self[ky] for ky in GEO_SHAPE[key]})
        return default

    def __setitem__(self, key: str, value: Union[float, complex, Geo]) -> None:
        """

        :param key:
        :param value:
        :return:
        """
        if key in self.keys() and isinstance(value, (float, int, bool)):
            self.__vec[key] = float(value)
        elif key in RED_GEO_SHAPE and isinstance(value, (complex,)):
            self.__vec[f"+{key}"] = value.real
            self.__vec[f"-{key}"] = value.imag
        elif key in ('-', '+') and isinstance(value, (Geo,)):
            for n_key in RED_GEO_SHAPE:
                self[key + n_key] = value[key + n_key]
        if key in GEO_SHAPE.keys() and isinstance(value, (Geo,)):
            for ky in GEO_SHAPE[key]:
                self[ky] = value[ky]
        raise KeyError(f"{key} does not exist in {self.keys()}.")

    def __delattr__(self, item) -> None:
        if item in self.keys():
            self.__vec[item] = 0.0
        else:
            del self.__vec[item]

    # ---- selective dimension-wise operations ------------
    def __format__(self, format_spec) -> Geo:
        reslt = Geo()
        for ky, val in self.items():
            reslt[ky] = format_spec(self[ky])
        return reslt

    def real(self) -> Geo:
        """
        The real elements of the Geo presented as scalar and vector values.
        :return:
        """
        nw_spc = Geo()
        for ky, val in self:
            if ky[0] == "+":
                nw_spc[ky] = val
        return nw_spc

    def imag(self) -> Geo:
        """
        The imaginary elements of the Geo presented as scalar and vector values.
        :return:
        """
        nw_spc = Geo()
        for ky, val in self:
            if ky[0] == "-":
                nw_spc['+' + ky[1]] = val
        return nw_spc

    def phase(self, deg=False) -> Geo:
        """
        Get the Geo showing the radian phase relative to the scalar.
        :return:
        """
        nw_spc = Geo()
        for ky in self.keys():
            if ky != '+0':
                c_num = self['+0'] + self[ky]*1j
                nw_spc[ky] = cm.phase(c_num)
                if deg:
                    nw_spc[ky] *= 180 / np.pi

        return nw_spc

    def mag_phase(self) -> (float, Geo):
        return self.magnitude(), self.phase()

    def conj(self) -> Geo:
        """
        The complex conjugate of the Geo.
        :return:
        """
        nw_spc = Geo()
        for ky, val in self:
            if ky[0] == "+":
                nw_spc[ky] = val
            else:
                nw_spc[ky] = -val
        return nw_spc

    def conjugate(self) -> Geo:
        return self.conj()

    def scale(self, other: Union[dict, float: 1.0, complex, int, bool, Geo]) -> Geo:
        """
        Get a scaled version of this Geo based on the dimensions of what was passed in.

        dict and Geo: dimension-wise scaling (in difference to x*y, this clears unspecified dims)
        float, int, bool: applied equally to all dims
        complex: real and imaginary parts are applied to '+' and '-' respectively

        :param other:
        :return:
        """
        nw_spc = Geo(self)
        if isinstance(other, (Geo, dict)):
            for ky in self.keys():
                if ky in other.keys():
                    nw_spc[ky] *= other[ky]
                else:
                    nw_spc[ky] = 0.0

        elif isinstance(other, (float, int, bool)):
            for ky in self.keys():
                nw_spc[ky] *= other
        else:
            for ky in self.keys():
                if ky[0] == '+':
                    nw_spc[ky] *= other.real
                elif ky[0] == '-':
                    nw_spc[ky] *= other.imag
        return nw_spc

    def __mul__(self, other: Union[dict, float: 1.0, complex, int, bool, Geo]) -> Geo:
        """
        perform a dimension-wise multiplication. Use the corresponding operator for
        inner, outer, and geometric products (&, ^, and | respectively). Use scale() for ... well, scaling.
        other can also be a sub-set of the full space.
        :param other:
        :return:
        """
        nw_spc = Geo(self)
        if isinstance(other, (Geo, dict)):
            for key in set(other.keys()).intersection(self.keys()):
                nw_spc[key] *= other[key]
        elif isinstance(other, (float, int, bool)):
            nw_spc[GEO_SHAPE["scalar"][0]] *= other
        else:
            nw_spc[GEO_SHAPE["scalar"][0]] *= other.real
            nw_spc[GEO_SHAPE["tri-vector"][0]] *= other.imag
        return nw_spc

    def __rmul__(self, other) -> Geo:
        return self * other

    def __pow__(self, power: Union[dict, float: 1.0, int, complex, bool], modulo=None) -> Geo:
        """
        Apply the power to each complex pair separately.
        :param power:
        :param modulo:
        :return:
        """
        geo_mag_vec = self.magnitude_vectorized()
        geo_phs_vec = self.phase_vectorized()

        reslt = Geo()

        for ky_num in RED_GEO_SHAPE:
            mag_val = geo_mag_vec['+' + ky_num] ** power
            phs_val = geo_phs_vec['-' + ky_num] * power
            reslt['+' + ky_num] += mag_val * np.cos(phs_val)
            reslt['-' + ky_num] += mag_val * np.sin(phs_val)

            return reslt

    def __rpow__(self, other) -> Geo:
        if not isinstance(other, Geo):
            other = Geo(other)
        return other ** self

    def __truediv__(self, other: Union[dict, float: 1.0, complex, int, bool, Geo]) -> Geo:
        """
        perform a dimension-wise multiplication. Use the corresponding operator for
        inner, outer, and geometric products (&, ^, and | respectively). Use one_over() to get 1/x.
        other can also be a sub-set of the full space.
        :param other:
        :return:
        """
        nw_spc = Geo(self)
        if isinstance(other, (Geo, dict)):
            for key in set(other.keys()).intersection(self.keys()):
                nw_spc[key] /= other[key]
        elif isinstance(other, (float, int, bool)):
            nw_spc[GEO_SHAPE["scalar"][0]] /= other
        else:
            nw_spc[GEO_SHAPE["scalar"][0]] /= other.real
            nw_spc[GEO_SHAPE["tri-vector"][0]] /= other.imag
        return nw_spc

    def __rtruediv__(self, other) -> Geo:
        if not isinstance(other, Geo):
            other = Geo(other, 0.0)
        return other / self

    def __floordiv__(self, other: Union[dict, float: 1.0, int, bool, Geo]) -> Geo:

        nw_spc = Geo(self)
        if isinstance(other, (Geo, dict)):
            for key in set(other.keys()).intersection(self.keys()):
                nw_spc[key] //= other[key]
        elif isinstance(other, (float, int, bool)):
            nw_spc[GEO_SHAPE["scalar"][0]] //= other
        else:
            nw_spc[GEO_SHAPE["scalar"][0]] //= other.real
            nw_spc[GEO_SHAPE["tri-vector"][0]] //= other.imag
        return nw_spc

    def __rfloordiv__(self, other) -> Geo:
        if not isinstance(other, Geo):
            other = Geo(other)
        return other // self

    def __divmod__(self, other: Union[dict, float: 1.0, int, bool, Geo]) -> (Geo, Geo):
        return self.__floordiv__(other=other), self.__mod__(other=other)

    def __rdivmod__(self, other):
        if not isinstance(other, Geo):
            other = Geo(other)
        return other.__divmod__(self)

    def __mod__(self, other: Union[dict, float: 1.0, int, bool, Geo]) -> Geo:
        return self / other - self.__floordiv__(other=other)

    def __rmod__(self, other) -> Geo:
        if not isinstance(other, Geo):
            other = Geo(other)
        return other % self

    def __add__(self, other: Union[dict, float: 0.0, complex, int, bool, Geo]) -> Geo:
        """
        other can also be a sub-set of the full space.
        :param other:
        :return:
        """
        nw_spc = Geo(self)
        if isinstance(other, (Geo, dict)):
            for key in set(other.keys()).intersection(self.keys()):
                nw_spc[key] += other[key]
        elif isinstance(other, (float, int, bool)):
            nw_spc[GEO_SHAPE["scalar"][0]] += other
        else:
            nw_spc[GEO_SHAPE["scalar"][0]] += other.real
            nw_spc[GEO_SHAPE["tri-vector"][0]] += other.imag
        return nw_spc

    def __radd__(self, other) -> Geo:
        return self + other

    def __sub__(self, other: Union[dict, float: 0.0, complex, int, bool, Geo]) -> Geo:
        """
        other can also be a sub-set of the full space.
        :param other:
        :return:
        """
        nw_spc = Geo(self)
        if isinstance(other, (Geo, dict)):
            for key in set(other.keys()).intersection(self.keys()):
                nw_spc[key] -= other[key]
        elif isinstance(other, (float, int, bool)):
            nw_spc[GEO_SHAPE["scalar"][0]] -= other
        else:
            nw_spc[GEO_SHAPE["scalar"][0]] -= other.real
            nw_spc[GEO_SHAPE["tri-vector"][0]] -= other.imag
        return nw_spc

    def __rsub__(self, other) -> Geo:
        return -self + other

    def __neg__(self) -> Geo:

        nw_spc = Geo()
        for key in self.keys():
            nw_spc[key] = -self[key]
        return nw_spc

    def __pos__(self) -> Geo:
        return Geo(self)

    def __abs__(self) -> Geo:
        """
        rectify the elements such that they are all positive. If you want a scalar, use magnitude
        :return:
        """
        n_self = Geo(src=self)

        for key in n_self.keys():
            n_self[key] = np.abs(n_self[key])
        return n_self

    def __ceil__(self) -> Geo:
        rslt = Geo()
        for ky, val in self.items():
            rslt[ky] = np.ceil(val)
        return rslt

    def __floor__(self) -> Geo:
        rslt = Geo()
        for ky, val in self.items():
            rslt[ky] = np.floor(val)
        return rslt

    def __round__(self) -> Geo:
        rslt = Geo()
        for ky, val in self.items():
            rslt[ky] = np.round(val)
        return rslt

    def __trunc__(self) -> Geo:
        rslt = Geo()
        for ky, val in self.items():
            rslt[ky] = np.trunc(val)
        return rslt

    def as_integer_ratio(self) -> (Geo, Geo):
        numer = Geo()
        denom = Geo()
        for ky, val in self.items():
            numer[ky], denom[ky] = np.as_integer_ratio(val)
        return numer, denom

    def copy(self) -> Geo:
        return Geo(self)

    # ---- cross-dimensional operations ------------
    def rot(self, other: Union[dict, float: 0.0, complex, int, bool, Geo]) -> Geo:
        """
        Use all elements to rotate other
        :param other:
        :return:
        """
        return self | other | -self.inverse().conj()

    def rot_imag(self, other: Union[dict, float: 0.0, complex, int, bool, Geo]) -> Geo:
        """
        use the real vector to rotate the imaginary elements of other.
        :param other:
        :return:
        """
        return self['+'] | other['-'] | -self['+'].inverse()

    def rot_real(self, other: Union[dict, float: 0.0, complex, int, bool, Geo]) -> Geo:
        """
        use the imaginary vector to rotate the real elements of other.
        :param other:
        :return:
        """
        return self['-'] | other['+'] | self['-'].inverse()

    def norm(self) -> Geo:
        mag = self.magnitude()
        if mag == 0.0:
            return self.copy()
        return self.scale(1/mag)

    def magnitude_vectorized(self) -> Geo:
        """
        Get the scalar and vector showing the magnitude of the complex values.
        :return:
        """
        nw_spc = Geo()
        for ky_num in RED_GEO_SHAPE:
            nw_spc['+' + ky_num] = np.sqrt(self['+' + ky_num] ** 2 + self['-' + ky_num] ** 2)

        return nw_spc

    def phase_vectorized(self, deg=False) -> Geo:
        """
        Get the bi-vector and tri-vector showing the radian phase of the complex values.
        :return:
        """
        nw_spc = Geo()
        for ky_num in RED_GEO_SHAPE:
            c_num = self['+' + ky_num] + self['-' + ky_num]*1j
            nw_spc['-' + ky_num] = cm.phase(c_num)
            if deg:
                nw_spc['-' + ky_num] *= 180 / np.pi

        return nw_spc

    def mag_pha_vectorized(self) -> (Geo, Geo):
        return self.magnitude_vectorized(), self.phase_vectorized()

    def inverse(self) -> Geo:
        """
        Get 1/x of this Geo.
        :return:
        """
        if self == 0:
            return Geo()
        nw_spc = self.conj()
        nw_spc /= self.magnitude_sq()
        return nw_spc

    def __and__(self, other: Union[dict, float: 0.0, complex, int, bool, Geo]) -> Geo:
        """
        other can also be a sub-set of the full space.
        This is equivalent to a vector inner(dot)-product
        :param other: contains a vector and/or bi-vector
        :return: a geospatial set with the resulting real scalar in +0
        """
        if not isinstance(other, Geo):
            other = Geo(other)

        nw_spc = Geo()
        for o_key in RED_GEO_SHAPE:
            p_key = '+' + o_key
            n_key = '-' + o_key
            # real results
            nw_spc[GEO_RED_KYS[n_key][n_key][0]] += self[n_key] * other[n_key] * GEO_RED_KYS[n_key][n_key][1]
            nw_spc[GEO_RED_KYS[p_key][p_key][0]] += self[p_key] * other[p_key] * GEO_RED_KYS[p_key][p_key][1]
            # imaginary results
            nw_spc[GEO_RED_KYS[p_key][n_key][0]] += self[p_key] * other[n_key] * GEO_RED_KYS[p_key][n_key][1]
            nw_spc[GEO_RED_KYS[n_key][p_key][0]] += self[n_key] * other[p_key] * GEO_RED_KYS[n_key][p_key][1]

        return nw_spc

    def __rand__(self, other) -> Geo:
        if not isinstance(other, Geo):
            other = Geo(other)
        return other & self

    def __xor__(self, other: Union[dict, float: 1.0, int, bool, Geo]) -> Geo:
        """
        This is equivalent to a vector outer-product.
        other can also be a sub-set of the full space.
        :param other: contains a vector and/or bi-vector
        :return: a geospatial set with the resulting vector in (+1, +2, +3)
        """

        if not isinstance(other, Geo):
            other = Geo(other)

        nw_spc = Geo()
        for o_key in RED_GEO_SHAPE_UPPER.keys():
            pk1 = '+' + o_key
            nk1 = '-' + o_key
            for s_key in RED_GEO_SHAPE_UPPER[o_key]:
                pk2 = '+' + s_key
                nk2 = '-' + s_key
                if s_key in ('1', '3') and o_key in ('1', '3'):
                    x_sgn = -1
                else:
                    x_sgn = 1
                # bi-vector and bi-vector cross-product
                nw_spc[GEO_RED_KYS[pk1][pk2][0]] += self[pk1] * other[pk2] * GEO_RED_KYS[pk1][pk2][1] * x_sgn
                nw_spc[GEO_RED_KYS[pk2][pk1][0]] -= self[pk2] * other[pk1] * GEO_RED_KYS[pk2][pk1][1] * x_sgn

                # bi-vector and tri-vector cross-product
                nw_spc[GEO_RED_KYS[pk1][nk2][0]] += self[pk1] * other[nk2] * GEO_RED_KYS[pk1][nk2][1] * x_sgn
                nw_spc[GEO_RED_KYS[pk2][nk1][0]] -= self[pk2] * other[nk1] * GEO_RED_KYS[pk2][nk1][1] * x_sgn

                # tri-vector and tri-vector cross-product
                nw_spc[GEO_RED_KYS[nk1][nk2][0]] += self[nk1] * other[nk2] * GEO_RED_KYS[nk1][nk2][1] * x_sgn
                nw_spc[GEO_RED_KYS[nk2][nk1][0]] -= self[nk2] * other[nk1] * GEO_RED_KYS[nk2][nk1][1] * x_sgn

                # tri-vector and bi-vector cross-product
                nw_spc[GEO_RED_KYS[nk1][pk2][0]] += self[nk1] * other[pk2] * GEO_RED_KYS[nk1][pk2][1] * x_sgn
                nw_spc[GEO_RED_KYS[nk2][pk1][0]] -= self[nk2] * other[pk1] * GEO_RED_KYS[nk2][pk1][1] * x_sgn
        # print(nw_spc)
        return nw_spc

    def __rxor__(self, other) -> Geo:
        if not isinstance(other, Geo):
            other = Geo(other)
        return other ^ self

    def __or__(self, other: Union[dict, float: 1.0, int, bool, Geo]) -> Geo:
        """
        other can also be a sub-set of the full space.
        This is equivalent to a vector geometric-product
        :param other: contains a scalar, vector, bi-vector, and/or tri-vector
        :return: a geospatial set
        """
        return (self & other) + (self ^ other)

    def __ror__(self, other) -> Geo:
        if not isinstance(other, Geo):
            other = Geo(other)
        return other | self

    def __matmul__(self, other: Union[dict, float: 1.0, int, bool, Geo]) -> Geo:
        """
        Rotation using the reference rotor provided as other.
        The rotor is not normalized here to allow for additional functionality.
        Remember that the rotation is twice the rotors phase as the geometric
        product is carried out twice.

        :param other: contains a scalar, vector, bi-vector, and/or tri-vector
        :return: a geospatial set
        """

        if not isinstance(other, Geo):
            other = Geo(other)
        # normy = other.norm()

        return other.inverse() | self | other

    def __rmatmul__(self, other) -> Geo:
        if not isinstance(other, Geo):
            other = Geo(other)
        return other @ self

    def __invert__(self) -> Geo:
        """
        swap the real and imaginary axes for each dimension.
        :return:
        """
        rslt = Geo()
        for ky1 in RED_GEO_SHAPE:
            rslt['+' + ky1] = rslt['-' + ky1]
            rslt['-' + ky1] = rslt['+' + ky1]

        return rslt

    def __lshift__(self, other: int) -> Geo:
        rslt = Geo(self)
        dim_lim = len(GEO_SHAPE["vector"])
        for ind in range(dim_lim):
            dst_ind = (ind - other) % dim_lim

            rslt[GEO_SHAPE["vector"][dst_ind]] = self[GEO_SHAPE["vector"][ind]]
            rslt[GEO_SHAPE["bi-vector"][dst_ind]] = self[GEO_SHAPE["bi-vector"][ind]]
        return rslt

    def __rshift__(self, other: int) -> Geo:
        rslt = Geo(self)
        dim_lim = len(GEO_SHAPE["vector"])
        for ind in range(dim_lim):
            dst_ind = (ind + other) % dim_lim

            rslt[GEO_SHAPE["vector"][dst_ind]] = self[GEO_SHAPE["vector"][ind]]
            rslt[GEO_SHAPE["bi-vector"][dst_ind]] = self[GEO_SHAPE["bi-vector"][ind]]
        return rslt

    # -------- comparison methods ---------
    def __lt__(self, other: Union[dict, float: 0.0, int, bool, Geo]) -> Union[int, Geo]:
        """ Less than
        Check against magnitude if it is being compared with a scalar.
        Convert the magnitude to match the scalar type.
        If a dict or Geo is passed in, return a Geo with per-dim boolean comparisons
        :param other:
        :return:
        """
        if isinstance(other, (float, int, bool)):
            return int(self.magnitude() < other)

        rslt = Geo()
        for ky in self.keys():
            if ky in other.keys():
                rslt[ky] = int(self[ky] < other[ky])
            else:
                rslt[ky] = int(self[ky] < 0)

        return rslt

    def __le__(self, other: Union[dict, float: 0.0, int, bool, Geo]) -> Union[int, Geo]:
        """ Less than or equal to
        Check against magnitude if it is being compared with a scalar.
        Convert the magnitude to match the scalar type.
        If a dict or Geo is passed in, return a Geo with per-dim boolean comparisons
        :param other:
        :return:
        """
        if isinstance(other, (float, int, bool)):
            return int(self.magnitude() <= other)

        rslt = Geo()
        for ky in self.keys():
            if ky in other.keys():
                rslt[ky] = int(self[ky] <= other[ky])
            else:
                rslt[ky] = int(self[ky] <= 0)

        return rslt

    def __eq__(self, other: Union[dict, float: 0.0, int, bool, Geo]) -> Union[int, Geo]:
        """ Equal to
        Check against magnitude if it is being compared with a scalar.
        Convert the magnitude to match the scalar type. We can then check if it is zero by using x == 0
        If a dict or Geo is passed in, return a Geo with per-dim boolean comparisons
        :param other:
        :return:
        """
        if isinstance(other, (float, int)):
            return int(self.magnitude() == other)
        elif isinstance(other, bool):
            return int(bool(self.magnitude()) == other)

        rslt = Geo()
        for ky in self.keys():
            if ky in other.keys():
                rslt[ky] = int(self[ky] == other[ky])
            else:
                rslt[ky] = int(self[ky] == 0)

        return rslt

    def __ne__(self, other: Union[dict, float: 0.0, int, bool, Geo]) -> Union[int, Geo]:
        """ Equal to
        Check against magnitude if it is being compared with a scalar.
        Convert the magnitude to match the scalar type. We can then check if it is zero by using x == 0
        If a dict or Geo is passed in, return a Geo with per-dim boolean comparisons
        :param other:
        :return:
        """
        if isinstance(other, (float, int)):
            return int(self.magnitude() != other)
        elif isinstance(other, bool):
            return int(bool(self.magnitude()) != other)

        rslt = Geo()
        for ky in self.keys():
            if ky in other.keys():
                rslt[ky] = int(self[ky] != other[ky])
            else:
                rslt[ky] = int(self[ky] != 0)

        return rslt

    def __ge__(self, other: Union[dict, float: 0.0, int, bool, Geo]) -> Union[int, Geo]:
        """ Greater than or equal to
        Check against magnitude if it is being compared with a scalar.
        Convert the magnitude to match the scalar type.
        If a dict or Geo is passed in, return a Geo with per-dim boolean comparisons
        :param other:
        :return:
        """
        if isinstance(other, (float, int, bool)):
            return int(self.magnitude() >= other)

        rslt = Geo()
        for ky in self.keys():
            if ky in other.keys():
                rslt[ky] = int(self[ky] >= other[ky])
            else:
                rslt[ky] = int(self[ky] >= 0)

        return rslt

    def __gt__(self, other: Union[dict, float: 0.0, int, bool, Geo]) -> Union[int, Geo]:
        """ Greater than
        Check against magnitude if it is being compared with a scalar.
        Convert the magnitude to match the scalar type.
        If a dict or Geo is passed in, return a Geo with per-dim boolean comparisons
        :param other:
        :return:
        """
        if isinstance(other, (float, int, bool)):
            return int(self.magnitude() >= other)

        rslt = Geo()
        for ky in self.keys():
            if ky in other.keys():
                rslt[ky] = int(self[ky] > other[ky])
            else:
                rslt[ky] = int(self[ky] > 0)

        return rslt

    def min(self, other: Union[dict, float: 0.0, complex, int, bool, Geo]) -> Geo:
        """
        return the smaller value between self and other
        :param other:
        :return:
        """

        rslt = self.copy()

        if isinstance(other, (float, int, bool)):
            if rslt[GEO_SHAPE['scalar'][0]] > other:
                rslt[GEO_SHAPE['scalar'][0]] = other
        elif isinstance(other, (complex,)):
            if rslt[GEO_SHAPE['scalar'][0]] > other.real:
                rslt[GEO_SHAPE['scalar'][0]] = other.real
            if rslt[GEO_SHAPE['tri-vector'][0]] > other.imag:
                rslt[GEO_SHAPE['tri-vector'][0]] = other.imag
        else:
            for ky in self.keys():
                if ky in other.keys() and other[ky] < self[ky]:
                    rslt[ky] = other[ky]

        return rslt

    def max(self, other: Union[dict, float: 0.0, complex, int, bool, Geo]) -> Geo:
        """
        return the larger value between self and other
        :param other:
        :return:
        """

        rslt = self.copy()

        if isinstance(other, (float, int, bool)):
            if rslt[GEO_SHAPE['scalar'][0]] < other:
                rslt[GEO_SHAPE['scalar'][0]] = other
        elif isinstance(other, (complex,)):
            if rslt[GEO_SHAPE['scalar'][0]] < other.real:
                rslt[GEO_SHAPE['scalar'][0]] = other.real
            if rslt[GEO_SHAPE['tri-vector'][0]] < other.imag:
                rslt[GEO_SHAPE['tri-vector'][0]] = other.imag
        else:
            for ky in self.keys():
                if ky in other.keys() and other[ky] > self[ky]:
                    rslt[ky] = other[ky]

        return rslt


def from_mag_pha_vectorized(mag_vec: Union[dict, float: 0.0, complex, int, bool, Geo],
                            pha_vec: Union[dict, float: 0.0, complex, int, bool, Geo]) -> Geo:
    if not isinstance(mag_vec, Geo):
        mag_vec = Geo(mag_vec)
    if not isinstance(pha_vec, Geo):
        pha_vec = Geo(pha_vec)

    nw_spc = Geo()
    for ky_num in RED_GEO_SHAPE:
        nw_spc['+' + ky_num] = mag_vec['+' + ky_num] * np.cos(pha_vec['-' + ky_num])
        nw_spc['-' + ky_num] = mag_vec['+' + ky_num] * np.sin(pha_vec['-' + ky_num])

    return nw_spc
