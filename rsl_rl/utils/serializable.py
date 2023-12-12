from typing import Any, Dict


class Serializable:
    def load_state_dict(self, data: Dict[str, Any]) -> None:
        """Loads agent parameters from a dictionary."""
        assert hasattr(self, "_serializable_objects")

        for name in self._serializable_objects:
            assert hasattr(self, name)
            assert name in data, f'Object "{name}" was not found while loading "{self.__class__.__name__}".'

            attr = getattr(self, name)
            if hasattr(attr, "load_state_dict"):
                print(f"Loading {name}")
                attr.load_state_dict(data[name])
            else:
                print(f"Loading value {name}={data[name]}")
                setattr(self, name, data[name])

    def state_dict(self) -> Dict[str, Any]:
        """Returns a dictionary containing the agent parameters."""
        assert hasattr(self, "_serializable_objects")

        data = {}

        for name in self._serializable_objects:
            assert hasattr(self, name)

            attr = getattr(self, name)
            data[name] = attr.state_dict() if hasattr(attr, "state_dict") else attr

        return data

    def _register_serializable(self, *objects) -> None:
        if not hasattr(self, "_serializable_objects"):
            self._serializable_objects = []

        for name in objects:
            if name in self._serializable_objects:
                continue

            self._serializable_objects.append(name)
