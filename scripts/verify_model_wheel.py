"""Join source, built wheel and offline-installed bytes for managed qualification.

Run after building and installing into a separate target directory. This script
never builds, installs or contacts a provider itself.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import sys
import zipfile
from pathlib import Path

PACKAGE = "amplifier_module_provider_github_copilot"


def main() -> None:
    source, dist, installed, output = map(Path, sys.argv[1:])
    wheels = list(dist.glob("*.whl"))
    assert len(wheels) == 1, wheels
    wheel = wheels[0]
    hashes: dict[str, str] = {}
    with zipfile.ZipFile(wheel) as archive:
        files = [name for name in archive.namelist() if name.startswith(PACKAGE + "/")]
        assert files
        for name in files:
            data = archive.read(name)
            assert data == (source / name).read_bytes(), ("source mismatch", name)
            assert data == (installed / name).read_bytes(), ("installed mismatch", name)
            hashes[name] = hashlib.sha256(data).hexdigest()
    import amplifier_module_provider_github_copilot as provider
    from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

    origin = Path(provider.__file__).resolve()
    assert origin.is_relative_to(installed.resolve()), origin
    assert not Path.cwd().is_relative_to(source.resolve())
    assert hasattr(GitHubCopilotProvider, "get_model_list_info")
    assert not hasattr(GitHubCopilotProvider, "discover_models")
    report = {
        "python": sys.executable,
        "origin": str(origin),
        "cwd": str(Path.cwd()),
        "wheel": wheel.name,
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "source_wheel_installed_sha256": hashes,
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("github-copilot-sdk", "amplifier-core", "pytest")
        },
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "source_wheel_installed_sha256"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
