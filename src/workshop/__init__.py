"""Workshop frontend over Latium's ROME, causal-tracing and GRAM implementations."""


def launch_workshop(*, work_dir=None, share=None):
    from src.workshop.app import launch_workshop as launch
    return launch(work_dir=work_dir, share=share)


__all__ = ["launch_workshop"]
