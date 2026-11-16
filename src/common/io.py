"""Result copies and processing workspaces outside the input directory."""

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import shutil
import tempfile


def write_json(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def copy_file(source, destination, records, overwrite=False):
    source, destination = Path(source), Path(destination)
    if not source.is_file():
        return False
    if destination.exists() and not overwrite:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    records.append(str(destination))
    return True


def copy_folder(source, destination, records, overwrite=False):
    source = Path(source)
    if source.is_dir():
        for path in sorted(source.rglob("*")):
            if path.is_file():
                copy_file(path, Path(destination) / path.relative_to(source), records, overwrite)


@contextmanager
def processing_workspace(project, kind):
    project.results.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".processing_{kind}_", dir=project.results) as folder:
        root = Path(folder).resolve()
        if not root.is_relative_to(project.results.resolve()):
            raise ValueError("Processing workspace is outside results.")
        (root / "code").mkdir()
        (root / "data").mkdir()
        yield root


@contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def input_inventory(project):
    return {str(p.relative_to(project.data)): [p.stat().st_size, p.stat().st_mtime_ns]
            for p in project.data.rglob("*") if p.is_file()}
