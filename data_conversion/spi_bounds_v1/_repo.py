"""Pfad-Helfer: SPI-Repo auffindbar machen (env SPI_REPO oder ../shadow-program-inversion)."""
import os
import sys

REPO = os.environ.get(
    "SPI_REPO",
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shadow-program-inversion")),
)
if REPO not in sys.path:
    sys.path.insert(0, REPO)
if not os.path.isdir(os.path.join(REPO, "shadow_program_inversion")):
    raise RuntimeError(
        f"SPI-Repo nicht gefunden unter {REPO}. Setze die Umgebungsvariable SPI_REPO "
        "auf den Pfad des geklonten shadow-program-inversion-Repos."
    )
