from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Union

SPATIAL_KEYS = ("+0", "-0", "+1", "-1", "+2", "-2", "+3", "-3")
GEO_SHAPE3 = ('0', '1', '2', '3')

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


def flip_sign(key1, key2) -> float:
    return GEO_ROT[key1[1] + key2[1]] * COM_ROT[key1[0] + key2[0]]


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

    dimension-wise operators:
    z = x * y => multiplication
    z = x / y => division
    z = x + y => addition
    z = x - y => subtraction

    !!!!!!!!!!!!!!!!!! USE BRACKETS TO ENSURE PROPER ORDER OF OPERATION !!!!!!!!!!!!!!!!!!!!!
    """

    def __init__(self, src: Union[dict, Geo] = None, def_val=0.0):
        """
        dimension 0 is assumed to be scalar while others are part of a vector/bi-/tri-.
        The layout is {+0, +1, +2, +3, -1, -2, -3, -0}
        """

        if src is None:
            self.__vec = {sp_key: def_val for sp_key in SPATIAL_KEYS}
        else:
            self.__vec = {sp_key: src.get(sp_key, def_val) for sp_key in SPATIAL_KEYS}

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
        return (self & self.conj())[SPATIAL_KEYS[0]]

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
        n_str = "<"
        for dim in SPATIAL_KEYS:
            n_str += f"{dim}: {self[dim]}, "

        n_str += ">"

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
        n_str = "Geo<"
        for dim in SPATIAL_KEYS:
            n_str += f"{dim}: {self[dim]}, "

        n_str += ">"

        return n_str

    # ----- dictionary/vector-like elements
    def keys(self):
        return self.__vec.keys()

    def values(self):
        return self.__vec.values()

    def items(self):
        return self.__vec.items()

    def __index__(self) -> int:
        return len(SPATIAL_KEYS)

    def __iter__(self) -> iter:
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

    def get(self, key, default) -> float:
        if key not in self.keys():
            return default
        return self[key]

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

    def subset(self, typ="scalar") -> Geo:
        """
        Get the sub-set Geo based on the global GEO_SHAPE.
        :param typ: 'scalar', 'vector', 'bi-vector', 'tri-vector'
        :return:
        """
        if typ in GEO_SHAPE.keys():
            return Geo(src={ky: self[ky] for ky in GEO_SHAPE[typ]})
        raise KeyError(f"{typ} is not a valid key for geometric subsets.")

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

    def __pow__(self, power: Union[dict, float: 1.0, int, complex, bool, Geo], modulo=None) -> Geo:
        """
        Apply the power to each axis separately.
        :param power:
        :param modulo:
        :return:
        """

        nw_spc = Geo(self)
        if isinstance(power, (Geo, dict)):
            for key in set(power.keys()).intersection(self.keys()):
                nw_spc[key] **= power[key]
        elif isinstance(power, (float, int, bool)):
            nw_spc[GEO_SHAPE["scalar"][0]] **= power
        else:
            nw_spc[GEO_SHAPE["scalar"][0]] **= power.real
            nw_spc[GEO_SHAPE["tri-vector"][0]] **= power.imag
        return nw_spc

    def __rpow__(self, other) -> Geo:
        if not isinstance(other, Geo):
            other = convert_to_geo(value=other)
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
            other = convert_to_geo(other, 0.0)
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
            other = convert_to_geo(other)
        return other // self

    def __divmod__(self, other: Union[dict, float: 1.0, int, bool, Geo]) -> (Geo, Geo):
        return self.__floordiv__(other=other), self.__mod__(other=other)

    def __rdivmod__(self, other):
        if not isinstance(other, Geo):
            other = convert_to_geo(value=other)
        return other.__divmod__(self)

    def __mod__(self, other: Union[dict, float: 1.0, int, bool, Geo]) -> Geo:
        return self / other - self.__floordiv__(other=other)

    def __rmod__(self, other) -> Geo:
        if not isinstance(other, Geo):
            other = convert_to_geo(other)
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

    def rotor(self, theta: Union[dict, float: 0.0, complex, int, bool, Geo]) -> Geo:
        return

    # ---- cross-dimensional operations ------------
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
        for ky_num in GEO_SHAPE3:
            nw_spc['+' + ky_num] = np.sqrt(self['+' + ky_num] ** 2 + self['-' + ky_num] ** 2)

        return nw_spc

    def phase_vectorized(self) -> Geo:
        """
        Get the scalar and vector showing the radian phase of the complex values.
        :return:
        """
        nw_spc = Geo()
        for ky_num in GEO_SHAPE3:
            c_num = self['+' + ky_num] + self['-' + ky_num]*1j
            nw_spc['+' + ky_num] = np.phase(c_num)

        return nw_spc

    def inverse(self) -> Geo:
        """
        Get 1/x of this Geo.
        :return:
        """
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
            other = convert_to_geo(value=other)

        nw_spc = Geo()
        for o_key in other.keys():
            nw_spc[GEO_SPATIAL_KEYS[o_key][o_key]] += self[o_key] * other[o_key] * COM_ROT[o_key[0] + o_key[0]]

        return nw_spc

    def __rand__(self, other) -> Geo:
        if not isinstance(other, Geo):
            other = convert_to_geo(value=other)
        return other & self

    def __xor__(self, other: Union[dict, float: 1.0, int, bool, Geo]) -> Geo:
        """
        This is equivalent to a vector outer-product.
        other can also be a sub-set of the full space.
        :param other: contains a vector and/or bi-vector
        :return: a geospatial set with the resulting vector in (+1, +2, +3)
        """

        if not isinstance(other, Geo):
            other = convert_to_geo(value=other)

        nw_spc = Geo()
        for o_key in other.keys():
            for s_key in self.keys():
                if s_key != o_key:
                    nw_spc[GEO_SPATIAL_KEYS[s_key][o_key]] += self[s_key] * other[o_key] * flip_sign(o_key, s_key) * 0.5

        return nw_spc

    def __rxor__(self, other) -> Geo:
        if not isinstance(other, Geo):
            other = convert_to_geo(value=other)
        return other ^ self

    def __or__(self, other: Union[dict, float: 1.0, int, bool, Geo]) -> Geo:
        """
        other can also be a sub-set of the full space.
        This is equivalent to a vector geometric-product
        :param other: contains a scalar, vector, bi-vector, and/or tri-vector
        :return: a geospatial set
        """

        if not isinstance(other, Geo):
            other = convert_to_geo(value=other)

        nw_spc = Geo()
        for o_key in other.keys():
            for s_key in self.keys():
                cmpnd = self[s_key] * other[o_key] * flip_sign(o_key, s_key)
                if o_key != s_key:
                    cmpnd *= 0.5

                nw_spc[GEO_SPATIAL_KEYS[s_key][o_key]] += cmpnd

        return nw_spc

    def __ror__(self, other) -> Geo:
        if not isinstance(other, Geo):
            other = convert_to_geo(value=other)
        return other | self

    def __matmul__(self, other: Union[dict, float: 1.0, int, bool, Geo]) -> Geo:
        """ todo implement this
        Rotation        (e**(-other.norm()*other.magnitude()/2)|self|e**(other.norm()*other.magnitude()/2)
        :param other: contains a scalar, vector, bi-vector, and/or tri-vector
        :return: a geospatial set
        """

        if not isinstance(other, Geo):
            other = convert_to_geo(value=other)
        normy = other.norm()

        return other.inverse() | self | other

    def __rmatmul__(self, other) -> Geo:
        if not isinstance(other, Geo):
            other = convert_to_geo(value=other)
        return other @ self

    def __invert__(self) -> Geo:
        """
        swap the real and imaginary axes for each dimension.
        :return:
        """
        rslt = Geo()
        for ky1 in GEO_SHAPE3:
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

    def pow_complex(self, power: Union[dict, float: 1.0, int, bool, Geo], modulo=None) -> Geo:
        """
        Apply the power to each complex pair separately then apply the geospatial transforms
        according to the geometric product.
        :param power:
        :param modulo:
        :return:
        """
        geo_mag_vec = self.magnitude_vectorized()
        geo_phs_vec = self.phase_vectorized()

        reslt = Geo()

        if isinstance(power, (float, int, bool)):
            for ky_num in GEO_SHAPE3:
                mag_val = geo_mag_vec['+' + ky_num] ** power
                phs_val = geo_phs_vec['-' + ky_num] * power
                reslt['+' + ky_num] += mag_val * np.cos(phs_val)
                reslt['-' + ky_num] += mag_val * np.sin(phs_val)

            return reslt

        # for p_key in power.keys(): todo this should cycle through different axes based on the power order
        #     for ky_num in GEO_SHAPE3:
        #         mag_val = geo_mag_vec['+' + ky_num] ** power[p_key]
        #         phs_val = geo_phs_vec['-' + ky_num] * power[p_key]
        #         reslt[GEO_SPATIAL_KEYS['+' + ky_num][p_key]] += mag_val * np.cos(phs_val) * flip_sign('+' + ky_num,
        #                                                                                               p_key)
        #         reslt[GEO_SPATIAL_KEYS['-' + ky_num][p_key]] += mag_val * np.sin(phs_val) * flip_sign('-' + ky_num,
        #                                                                                               p_key)
        raise ValueError('geo-powers is not yet implemented')
        # return reslt

    def pow_geo(self, power: Union[dict, float: 1.0, int, bool, Geo], modulo=None) -> Geo:

        geo_mag_vec = self.magnitude_vectorized()
        geo_phs_vec = self.phase_vectorized()

        reslt = Geo()

        if isinstance(power, (float, int, bool)):
            for ky_num in GEO_SHAPE3:
                mag_val = geo_mag_vec['+' + ky_num] ** power
                phs_val = geo_phs_vec['-' + ky_num] * power
                reslt['+' + ky_num] += mag_val * np.cos(phs_val)
                reslt['-' + ky_num] += mag_val * np.sin(phs_val)

            return reslt

        # for p_key in power.keys(): todo this should cycle through different axes based on the power order
        #     for ky_num in GEO_SHAPE3:
        #         mag_val = geo_mag_vec['+' + ky_num] ** power[p_key]
        #         phs_val = geo_phs_vec['-' + ky_num] * power[p_key]
        #         reslt[GEO_SPATIAL_KEYS['+' + ky_num][p_key]] += mag_val * np.cos(phs_val) * flip_sign('+' + ky_num,
        #                                                                                               p_key)
        #         reslt[GEO_SPATIAL_KEYS['-' + ky_num][p_key]] += mag_val * np.sin(phs_val) * flip_sign('-' + ky_num,
        #                                                                                               p_key)
        raise ValueError('geo-powers is not yet implemented')
        # return reslt

    # ------- et al. ------------
    # -------- comparison methods ---------
    def __lt__(self, other: Union[dict, float: 0.0, int, bool, Geo]) -> Union[bool, Geo]:
        """ Less than
        Check against magnitude if it is being compared with a scalar.
        Convert the magnitude to match the scalar type.
        If a dict or Geo is passed in, return a Geo with per-dim boolean comparisons
        :param other:
        :return:
        """
        if isinstance(other, (float, int, bool)):
            return self.magnitude() < other

        rslt = Geo()
        for ky in self.keys():
            if ky in other.keys():
                rslt[ky] = self[ky] < other[ky]
            else:
                rslt[ky] = self[ky] < 0

        return rslt

    def __le__(self, other: Union[dict, float: 0.0, int, bool, Geo]) -> Union[bool, Geo]:
        """ Less than or equal to
        Check against magnitude if it is being compared with a scalar.
        Convert the magnitude to match the scalar type.
        If a dict or Geo is passed in, return a Geo with per-dim boolean comparisons
        :param other:
        :return:
        """
        if isinstance(other, (float, int, bool)):
            return self.magnitude() <= other

        rslt = Geo()
        for ky in self.keys():
            if ky in other.keys():
                rslt[ky] = self[ky] <= other[ky]
            else:
                rslt[ky] = self[ky] <= 0

        return rslt

    def __eq__(self, other: Union[dict, float: 0.0, int, bool, Geo]) -> Union[bool, Geo]:
        """ Equal to
        Check against magnitude if it is being compared with a scalar.
        Convert the magnitude to match the scalar type. We can then check if it is zero by using x == 0
        If a dict or Geo is passed in, return a Geo with per-dim boolean comparisons
        :param other:
        :return:
        """
        if isinstance(other, (float, int)):
            return self.magnitude() == other
        elif isinstance(other, bool):
            return bool(self.magnitude()) == other

        rslt = Geo()
        for ky in self.keys():
            if ky in other.keys():
                rslt[ky] = self[ky] == other[ky]
            else:
                rslt[ky] = self[ky] == 0

        return rslt

    def __ne__(self, other: Union[dict, float: 0.0, int, bool, Geo]) -> Union[bool, Geo]:
        """ Equal to
        Check against magnitude if it is being compared with a scalar.
        Convert the magnitude to match the scalar type. We can then check if it is zero by using x == 0
        If a dict or Geo is passed in, return a Geo with per-dim boolean comparisons
        :param other:
        :return:
        """
        if isinstance(other, (float, int)):
            return self.magnitude() != other
        elif isinstance(other, bool):
            return bool(self.magnitude()) != other

        rslt = Geo()
        for ky in self.keys():
            if ky in other.keys():
                rslt[ky] = self[ky] != other[ky]
            else:
                rslt[ky] = self[ky] != 0

        return rslt

    def __ge__(self, other: Union[dict, float: 0.0, int, bool, Geo]) -> Union[bool, Geo]:
        """ Greater than or equal to
        Check against magnitude if it is being compared with a scalar.
        Convert the magnitude to match the scalar type.
        If a dict or Geo is passed in, return a Geo with per-dim boolean comparisons
        :param other:
        :return:
        """
        if isinstance(other, (float, int, bool)):
            return self.magnitude() >= other

        rslt = Geo()
        for ky in self.keys():
            if ky in other.keys():
                rslt[ky] = self[ky] >= other[ky]
            else:
                rslt[ky] = self[ky] >= 0

        return rslt

    def __gt__(self, other: Union[dict, float: 0.0, int, bool, Geo]) -> Union[bool, Geo]:
        """ Greater than
        Check against magnitude if it is being compared with a scalar.
        Convert the magnitude to match the scalar type.
        If a dict or Geo is passed in, return a Geo with per-dim boolean comparisons
        :param other:
        :return:
        """
        if isinstance(other, (float, int, bool)):
            return self.magnitude() >= other

        rslt = Geo()
        for ky in self.keys():
            if ky in other.keys():
                rslt[ky] = self[ky] > other[ky]
            else:
                rslt[ky] = self[ky] > 0

        return rslt


def convert_to_geo(value: Union[float, int, bool, complex, dict], def_val=0.0) -> Geo:
    if isinstance(value, dict):
        return Geo(value, def_val=def_val)

    rslt = Geo(def_val=def_val)
    if isinstance(value, (float, int, bool)):
        rslt[GEO_SHAPE["scalar"][0]] = value
    else:
        rslt[GEO_SHAPE["scalar"][0]] = value.real
        rslt[GEO_SHAPE["tri-vector"][0]] = value.imag

    return rslt