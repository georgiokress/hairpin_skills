# Offline-Exporter

Dieser zusätzliche Exporter ändert weder Controller noch Detektor, Rohbags oder historische Auswertungen. Er startet kein Training und sendet keine ROS-Service-, Aktions- oder Bewegungsanforderungen. Eingabe ist ausschließlich die geprüfte Liste von 90 abgeschlossenen Hauptserienläufen; zehn technische Abbruchansätze bleiben separat.

## Reproduktion in einem neuen Ausgabeordner

1. System-Python 3.12 mit ROS Jazzy: `extract_native.py INVENTORY NEW_OUTPUT_DIR` nach Laden von `/opt/ros/jazzy/setup.bash`.
2. Dieselbe ROS-Umgebung: `reconcile_native.py NEW_OUTPUT_DIR`. Fehlende ergänzende Kontexte können nur bei vollständigen Messserien und exakt belegbarer Planzuordnung aufgelöst werden. Ursprüngliche Ausschluss-/Exportregister werden erhalten.
3. In der aktivierten SPI-Venv: `analyze_export.py NEW_OUTPUT_DIR`.

Die historischen absoluten Pfade wurden für die Übergabe durch Umgebungsvariablen
ersetzt: `HAIRPIN_MAIN_CONFIG_ROOT`, `HAIRPIN_MAIN_RESULTS_ROOT`,
`HAIRPIN_MAIN_BAG_ROOT` und `HAIRPIN_MAIN_RUN_PLAN`.

Der Ausgabeordner muss neu sein; vorhandene Exporte werden nicht überschrieben. Die Bibliothekskopien in `spi_bounds_v1/` sind Bestandteil dieses Exporters. Ihre Hashes stehen in `source_code_hashes.json`. Der Detektorhash ist fest geprüft; andere Konfigurationen benötigen eine eigene Version.

Tests ohne Roboter: mit der SPI-Venv `python -m unittest discover -s CODE_DIR -p test_export.py -v`. Diese testen gruppierte Mitgliedschaften, EOS, native Referenzkorrektur, fehlende Detektion, Masken, Padding und das Laden mit dem bestehenden SPI-Loader. Unbekannte Slip-Annotationen werden im Rohdatenextraktor abgewiesen; Nutzerannahmen bleiben als Annahmen erkennbar.

## Datenvertrag und Grenzen

- Alle 90 Hauptserienläufe sind Development. CV-Heldout ist kein unabhängiger Modelltest.
- Inputs sind nominale Geschwindigkeit in m/s und positiver Kraftbetrag in N; reale Geschwindigkeit steht separat.
- Trajektorie 640×15 bei 32 ms: EOS, Success, Position, Quaternion wxyz, sechs interne Wrench-Kanäle.
- Native z_R-Detektion vor dem Resampling, eingefrorene Schwelle 0,075 mm und Persistenz/EMA 32 ms. Ortsfehler gegen laufrelatives r_GT.
- Passive ATI separat, nicht als Modellinput. Sensorachsen rotiert, Momentenbezugspunkt nicht verschoben.
- Paddingmaske und gültige Länge separat im HDF.

Die maschinenlesbaren CSV-Dateien sind flache Interchange-Tabellen aus den analytischen JSON-Ergebnissen, keine formatierten Excel-Arbeitsmappen. Eine Zeile je Lauf bzw. Parameterkombination, Einheiten im Feldnamen, unbekannte Werte als leer/null und Annahmenfelder separat.
