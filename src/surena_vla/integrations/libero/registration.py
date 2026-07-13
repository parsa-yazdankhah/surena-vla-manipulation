"""Runtime registration of the complete validated SURENA task catalog."""

from __future__ import annotations

def register_surena_tasks(task_mapping=None):
    from .surena_manipulation import register_surena_tasks as _register
    return _register(task_mapping)
