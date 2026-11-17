# SPI-Offline-Code

Dieser Ordner enthält den für die Abschlussarbeit verwendeten eigenständigen
PyTorch-Adapter: Datenprüfung, maskiertes Residual-GRU, Endpoint-Vertrag,
Training, Development-Refit und differentiable Multi-Start-Inversion.

Die Skripte bewegen keinen Roboter und greifen nicht auf ROS-Controller zu.
Ein vollständiger Ablauf mit Befehlen und erwarteter Verzeichnisstruktur steht
in [`../docs/SPI_WORKFLOW.md`](../docs/SPI_WORKFLOW.md).

Konfigurierbare Pfade:

- `HAIRPIN_DATASET_ROOT`: enthält `main_doe_v1_spi_v1/` und
  `development_main90_pilot27_v1/`;
- `HAIRPIN_RESULTS_ROOT`: Zielwurzel der Trainingsergebnisse.

Ohne Variablen werden `datasets/phase_b_spi` und `results/phase_b_spi`
relativ zum aktuellen Arbeitsverzeichnis verwendet.
