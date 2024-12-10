from pathlib import Path


def should_save(path: str, root: str) -> bool:
    relative_path = Path(path).relative_to(root)

    return Path(path).suffix == '.py' and (
        relative_path.parts[:2] == ('src', 'continuousp') or relative_path.parts[:1] == ('exec',)
    )
