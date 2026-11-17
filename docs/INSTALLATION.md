# Installation und Reproduzierbarkeit

## 1. Voraussetzungen

Der dokumentierte Stand wurde mit Ubuntu 24.04, ROS 2 Jazzy und Python 3.12
verwendet. Benötigt werden außerdem `colcon`, `rosdep` und `vcstool`.

```bash
sudo apt update
sudo apt install \
  ros-jazzy-desktop \
  python3-colcon-common-extensions \
  python3-rosdep \
  python3-vcstool
```

Falls `rosdep` auf dem Rechner noch nicht eingerichtet wurde:

```bash
sudo rosdep init
rosdep update
```

## 2. Workspace anlegen

```bash
mkdir -p ~/hairpin_ws/src
cd ~/hairpin_ws/src
git clone https://github.com/georgiokress/hairpin_skills.git
vcs import . < hairpin_skills/hairpin_skills.repos
```

`hairpin_skills.repos` pinnt den Universal-Robots-Treiber auf den in der
Arbeit verwendeten Tag. Der finale Aufnahmeweg verwendet ausschließlich den
internen UR-F/T-Sensor zur Regelung und benötigt deshalb keinen lokalen
Treiberpatch.

## 3. Abhängigkeiten installieren und bauen

```bash
cd ~/hairpin_ws
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -r -y --rosdistro jazzy
colcon build --symlink-install
source install/setup.bash
```

Die ROS-Abhängigkeiten werden über `package.xml` und `rosdep` beschrieben.
Die davon getrennte Offline-SPI-Umgebung wird anschließend mit
`spi/requirements.txt` installiert; Details stehen in `SPI_WORKFLOW.md`.

## 4. Installation prüfen

Diese Befehle starten keine Roboterbewegung:

```bash
cd ~/hairpin_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash

ros2 pkg executables hairpin_skills
ros2 launch hairpin_skills ur3e_bringup.launch.py --show-args
colcon test --packages-select hairpin_skills
colcon test-result --verbose
```

Erwartet werden genau die beiden Abschluss-Entry-Points `ati_rdt_wrench` und
`phase_b_spi_data_collection`. `radial_sliding_test.py` bleibt als interne
Basisklasse des Collectors installiert, ist aber kein eigener Bedienbefehl.

Der Paket-Build und die auf den ausführbaren ROS-Abschlusscode begrenzten
ament-Linttests sind für den dokumentierten Stand verifiziert. Die getrennten
Offline-SPI-Tests laufen in der in `SPI_WORKFLOW.md` beschriebenen Umgebung.

Für den bewegungsfreien Manifest-Validator:

```bash
python3 -m pip install -r ~/hairpin_ws/src/hairpin_skills/tools/requirements.txt
python3 ~/hairpin_ws/src/hairpin_skills/tools/validate_run_manifest.py --help
```

## 5. Aufbauabhängige Dateien

Folgende Dateien sind keine allgemeinen Standardwerte, sondern Bestandteil
des validierten Laboraufbaus:

- `config/ur3e_calibration.yaml`: roboterspezifische kinematische Kalibrierung;
- `launch/ur3e_bringup.launch.py`: IPs, Sensor-Frames und `sliding_frame`;
- `config/phase_b_spi_data_collection.yaml`: Kraft, Geschwindigkeit, Pfade und
  Aufnahmegates;
- `config/rosbag_qos_overrides.yaml`: QoS für `/tf` und `/tf_static`.
- `schemas/run_manifest.schema.json`: versioniertes Schema jedes Runs;
- `examples/rgt_caliper_v1.json`: dokumentierte geometrische Referenz des
  verwendeten Setznest2-Aufbaus.

Paketressourcen werden installationsrelativ gefunden. Rohdaten landen
standardmäßig unter `~/hairpin_data/bags`; alternativ kann vor dem Start
`HAIRPIN_DATA_ROOT` gesetzt oder `bag_root` als ROS-Parameter überschrieben
werden. Bei einer Installation ohne `--symlink-install` kann zusätzlich
`HAIRPIN_REPOSITORY_PATH` auf den Git-Clone gesetzt werden, damit dessen
Commit im Run-Manifest landet. Vor der Nutzung an einem anderen Roboter oder Aufbau
müssen Kalibrierung, TCP, Payload, Startpose, Wegpunkte und Prozessframe neu
validiert werden.


