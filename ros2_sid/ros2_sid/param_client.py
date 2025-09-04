#!/usr/bin/env python3
from typing import Optional, List

from rcl_interfaces.msg import Parameter as ParameterMsg, ParameterValue, ParameterType


def make_param(name: str, value:Optional) -> ParameterMsg:
    pv = ParameterValue()
    if isinstance(value, bool):
        pv.type = ParameterType.PARAMETER_BOOL
        pv.bool_value = value
    elif isinstance(value, int):
        pv.type = ParameterType.PARAMETER_INTEGER
        pv.integer_value = value
    elif isinstance(value, float):
        pv.type = ParameterType.PARAMETER_DOUBLE
        pv.double_value = float(value)
    elif isinstance(value, str):
        pv.type = ParameterType.PARAMETER_STRING
        pv.string_value = value
    else:
        raise ValueError(f"Unsupported type for {name}: {type(value)}")

    p = ParameterMsg()
    p.name = name
    p.value = pv
    return p

def pv_to_py(pv: ParameterValue):
    t = pv.type
    if t == ParameterType.PARAMETER_BOOL:   return pv.bool_value
    if t == ParameterType.PARAMETER_INTEGER:return pv.integer_value
    if t == ParameterType.PARAMETER_DOUBLE: return pv.double_value
    if t == ParameterType.PARAMETER_STRING: return pv.string_value
    if t == ParameterType.PARAMETER_BYTE_ARRAY:   return list(pv.byte_array_value)
    if t == ParameterType.PARAMETER_BOOL_ARRAY:   return list(pv.bool_array_value)
    if t == ParameterType.PARAMETER_INTEGER_ARRAY:return list(pv.integer_array_value)
    if t == ParameterType.PARAMETER_DOUBLE_ARRAY: return list(pv.double_array_value)
    if t == ParameterType.PARAMETER_STRING_ARRAY: return list(pv.string_array_value)
    return None

