from typing import Any
from collections.abc import Iterable, Mapping
from dataclasses import MISSING


def update_class_from_dict(obj, data: dict[str, Any], _ns: str = "") -> None:
    """Reads a dictionary and sets object variables recursively.

    This function performs in-place update of the class member attributes.

    Args:
        obj: An instance of a class to update.
        data: Input dictionary to update from.
        _ns: Namespace of the current object. This is useful for nested configuration
            classes or dictionaries. Defaults to "".

    Raises:
        TypeError: When input is not a dictionary.
        ValueError: When dictionary has a value that does not match default config type.
        KeyError: When dictionary has a key that does not exist in the default config type.
    """
    from isaaclab.utils.string import string_to_callable

    for key, value in data.items():
        # key_ns is the full namespace of the key
        key_ns = _ns + "/" + key
        # check if key is present in the object
        if hasattr(obj, key) or isinstance(obj, dict):
            obj_mem = obj[key] if isinstance(obj, dict) else getattr(obj, key)
            if isinstance(value, Mapping):
                # recursively call if it is a dictionary
                update_class_from_dict(obj_mem, value, _ns=key_ns)
                continue
            if isinstance(value, Iterable) and not isinstance(value, str):
                # check length of value to be safe
                if len(obj_mem) != len(value) and obj_mem is not None:
                    raise ValueError(
                        f"[Config]: Incorrect length under namespace: {key_ns}."
                        f" Expected: {len(obj_mem)}, Received: {len(value)}."
                    )
                if isinstance(obj_mem, tuple):
                    for i, (v, o) in enumerate(zip(value, obj_mem)):
                        if isinstance(v, int) and isinstance(o, float):
                            value[i] = float(value[i])
                    value = tuple(value)
                else:
                    set_obj = True
                    # recursively call if iterable contains dictionaries
                    for i in range(len(obj_mem)):
                        if isinstance(value[i], dict):
                            update_class_from_dict(obj_mem[i], value[i], _ns=key_ns)
                            set_obj = False
                    # do not set value to obj, otherwise it overwrites the cfg class with the dict
                    if not set_obj:
                        continue
            elif callable(obj_mem):
                # update function name
                value = string_to_callable(value)
            elif isinstance(value, int) and isinstance(obj_mem, float):
                value = float(value)
            elif isinstance(value, type(obj_mem)) or value is None or obj_mem == MISSING or obj_mem is None:
                pass
            else:
                raise ValueError(
                    f"[Config]: Incorrect type under namespace: {key_ns}."
                    f" Expected: {type(obj_mem)}, Received: {type(value)}."
                )
            # set value
            if isinstance(obj, dict):
                obj[key] = value
            else:
                setattr(obj, key, value)
        elif value is not None:
            raise KeyError(f"[Config]: Key not found under namespace: {key_ns}.")
