"""Workshop frontend over Latium's ROME, causal-tracing and GRAM implementations."""


def launch_workshop(*, work_dir=None, share=None, seed=42, covariance_drive_url=None):
    from src.workshop.app import launch_workshop as launch
    return launch(work_dir=work_dir, share=share, seed=seed, covariance_drive_url=covariance_drive_url)


__all__ = ["launch_workshop"]
