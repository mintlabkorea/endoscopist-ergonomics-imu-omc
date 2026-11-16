"""Project paths, input discovery, and optional procedure time limits."""

from dataclasses import dataclass
from pathlib import Path
import json
import math
import re


OMC_TRIAL_ID = re.compile(
    r"g_(?P<participant>[^_]+)_(?P<monitor>L|M|H)_(?P<bed>VL|L|M)_(?P<repeat>[1-9]\d*)"
)


def parse_omc_trial_id(name):
    """Read participant, workstation condition, and repetition from an OMC trial ID."""
    match = OMC_TRIAL_ID.fullmatch(name)
    if match is None:
        raise ValueError("OMC trial IDs must follow g_<participant>_<monitor>_<bed>_<repeat>.")
    return match.groupdict()


def natural_key(value):
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", str(value))]


@dataclass(frozen=True)
class Project:
    root: Path

    def __post_init__(self):
        object.__setattr__(self, "root", Path(self.root).resolve())

    @classmethod
    def default(cls):
        return cls(Path(__file__).resolve().parents[1])

    @property
    def data(self):
        return self.root / "data"

    @property
    def results(self):
        return self.root / "results"

    @property
    def time_config(self):
        return self.root / "config" / "imu_valid_time.json"

    def input_ids(self, kind):
        folder = self.data / kind
        if not folder.is_dir():
            return []
        if kind == "omc":
            names = [p.stem for p in folder.glob("g_*.csv")
                     if OMC_TRIAL_ID.fullmatch(p.stem)]
        elif kind == "validation":
            names = [name for name in ("neck", "shoulder", "elbow", "wrist")
                     if (folder / name).is_dir()]
        else:
            names = [p.name for p in folder.iterdir()
                     if p.is_dir() and re.fullmatch(r"p\d+_\d+", p.name)]
        return sorted(names, key=natural_key)

    def selected_ids(self, kind, requested=None):
        available = self.input_ids(kind)
        if not available:
            raise FileNotFoundError(f"No {kind} inputs found under {self.data / kind}")
        if not requested:
            return available
        selected = list(dict.fromkeys(requested))
        missing = sorted(set(selected) - set(available), key=natural_key)
        if missing:
            raise ValueError(f"Unknown {kind} input IDs: {', '.join(missing)}")
        return selected

    def _configuration(self):
        if not self.time_config.is_file():
            return {}
        value = json.loads(self.time_config.read_text(encoding="utf-8-sig"))
        if not isinstance(value, dict):
            raise ValueError("imu_valid_time.json must contain an object keyed by procedure ID.")
        return value

    def time_limits(self):
        value = {key: bounds for key, bounds in self._configuration().items() if key != "clinical_summary"}
        for key, bounds in value.items():
            if not re.fullmatch(r"p\d+_\d+", key):
                raise ValueError(f"Invalid procedure ID in time configuration: {key}")
            if (not isinstance(bounds, list) or len(bounds) != 2
                    or any(isinstance(v, bool) or not isinstance(v, (int, float))
                           or not math.isfinite(v) for v in bounds)
                    or bounds[0] > bounds[1]):
                raise ValueError(f"Invalid time range for {key}; expected [start, end].")
        return value

    def clinical_summary_options(self):
        options = self._configuration().get("clinical_summary", {})
        if not isinstance(options, dict):
            raise ValueError("clinical_summary configuration must be an object.")
        return options

    def output(self, *parts):
        path = self.results.joinpath(*parts).resolve()
        if not path.is_relative_to(self.results.resolve()):
            raise ValueError(f"Output path leaves the results directory: {path}")
        if path.is_relative_to(self.data.resolve()):
            raise ValueError("The input data directory cannot be an output directory.")
        return path
