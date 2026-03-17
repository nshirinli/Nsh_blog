from __future__ import annotations

import copy
import json
from pathlib import Path

_DEFAULTS: dict = {
    "units": {
        "pressure":    "Pa",
        "temperature": "K",
        "volume":      "L",
        "energy":      "J",
        "flow_rate":   "mol/s",
        "mass":        "kg",
    },
    "solver": {
        "ode_method": "RK45",
        "rtol":       1e-6,
        "atol":       1e-8,
        "max_steps":  10000,
    },
    "plot": {
        "dpi":      100,
        "grid":     True,
        "colormap": "default",
    },
    "appearance": {
        "theme":     "light",
        "accent":    "blue",
        "font_size": 13,
    },
}

_SETTINGS_FILE = Path.home() / ".chemeng_app" / "settings.json"


class SettingsManager:
    _instance: "SettingsManager | None" = None

    def __new__(cls) -> "SettingsManager":
        if cls._instance is None:
            obj = super().__new__(cls)
            obj._data: dict = {}
            obj._load()
            cls._instance = obj
        return cls._instance

    def _load(self) -> None:
        self._data = copy.deepcopy(_DEFAULTS)
        if _SETTINGS_FILE.exists():
            try:
                with open(_SETTINGS_FILE) as f:
                    saved = json.load(f)
                for section, vals in saved.items():
                    if section in self._data and isinstance(vals, dict):
                        self._data[section].update(vals)
            except Exception:
                pass  # corrupt / unreadable — use defaults

    def save(self) -> None:
        _SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_SETTINGS_FILE, "w") as f:
            json.dump(self._data, f, indent=2)

    def reset(self) -> None:
        self._data = copy.deepcopy(_DEFAULTS)
        self.save()

    def get(self, section: str, key: str):
        return self._data.get(section, {}).get(key)

    def set(self, section: str, key: str, value) -> None:
        self._data.setdefault(section, {})[key] = value

    def section(self, name: str) -> dict:
        return dict(self._data.get(name, {}))


# Module-level singleton
settings = SettingsManager()
