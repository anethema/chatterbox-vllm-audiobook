from collections.abc import Callable
from pathlib import Path


StaticPathRegistrar = Callable[..., None]


def register_completed_audiobook(
    output_path: str | Path,
    output_root: str | Path,
    register_static_paths: StaticPathRegistrar,
) -> Path:
    """Expose one completed M4B without exposing its project directory.

    Gradio normally copies output files into its temporary cache. Registering the
    exact final file as static makes Gradio serve it from its durable project
    location instead. Call this only after final verification and project cleanup.
    """
    root = Path(output_root).resolve()
    completed = Path(output_path).resolve()

    try:
        completed.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"Completed audiobook must be inside the output root: {completed}"
        ) from error

    if completed.suffix.lower() != ".m4b":
        raise ValueError(f"Completed audiobook must be an M4B file: {completed}")
    if not completed.is_file():
        raise FileNotFoundError(f"Completed audiobook does not exist: {completed}")

    # Register only the final file. Registering its parent would also expose the
    # saved source EPUB and private reference-voice sample.
    register_static_paths(paths=[completed])
    return completed
