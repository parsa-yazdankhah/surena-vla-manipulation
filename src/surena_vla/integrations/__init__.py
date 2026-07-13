"""Optional integrations are registered explicitly to avoid import side effects."""


def register_all() -> None:
    from .robosuite import register_surena_robot

    register_surena_robot()

    from .libero import register_surena_tasks

    register_surena_tasks()


__all__ = ["register_all"]
