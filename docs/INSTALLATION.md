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
Arbeit verwendeten Tag. Für den exakten historischen Treiberstand wird danach
der mitgelieferte Patch angewendet:

```bash
cd ~/hairpin_ws/src/Universal_Robots_ROS2_Driver
git apply --check ../hairpin_skills/patches/ur_driver_external_ft.patch
git apply ../hairpin_skills/patches/ur_driver_external_ft.patch
```

Der Patch ergänzt eine optionale externe F/T-RTDE-Aktivierung. Der finale
Phase-B-Aufnahmeweg setzt diese Funktion ausdrücklich auf `false`; der Patch
wird trotzdem mitgeführt, damit der verwendete Workspace vollständig
rekonstruierbar bleibt.

## 3. Abhängigkeiten installieren und bauen

```bash
cd ~/hairpin_ws
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -r -y --rosdistro jazzy
colcon build --symlink-install
source install/setup.bash
```

Die Python-Abhängigkeiten werden über `package.xml` und `rosdep` beschrieben.
Ein separates `requirements.txt` wäre für dieses ROS-Paket daher redundant.

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

Erwartet werden unter anderem die Entry-Points `ati_rdt_wrench`,
`phase_a_radial_ik_validator`, `phase_b_spi_data_collection` und
`phase_c_chamfer_scan`.

Der Paket-Build ist für den dokumentierten Stand verifiziert. Die vorhandenen
ament-Linttests melden allerdings bestehende Flake8-/Pep257-Verstöße im
Forschungsquellcode. Ein rotes Lintergebnis ist transparent zu dokumentieren;
es darf weder als erfolgreicher Test ausgegeben noch mit einer realen
Hardwarevalidierung verwechselt werden.

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
- `config/phase_a_radial_ik_validator.yaml`: Startpose und IK-Wegpunktdatei;
- `config/phase_b_spi_data_collection.yaml`: Kraft, Geschwindigkeit, Pfade und
  Aufnahmegates;
- `config/rosbag_qos_overrides.yaml`: QoS für `/tf` und `/tf_static`.
- `schemas/run_manifest.schema.json`: versioniertes Schema jedes Runs;
- `examples/rgt_caliper_v1.json`: dokumentierte geometrische Referenz des
  verwendeten Setznest2-Aufbaus.

Vor der Nutzung auf einem anderen PC müssen die absoluten Pfade unter
`/home/georgio/masterarbeit` entweder angelegt oder als ROS-Parameter
überschrieben werden. Vor der Nutzung an einem anderen Roboter oder Aufbau
müssen Kalibrierung, TCP, Payload, Startpose, Wegpunkte und Prozessframe neu
validiert werden.

## 6. Versionsnachweis

Für eine wissenschaftlich nachvollziehbare Aufnahme sollten vor dem Versuch
mindestens folgende Informationen gesichert werden:

```bash
git -C ~/hairpin_ws/src/hairpin_skills rev-parse HEAD
git -C ~/hairpin_ws/src/hairpin_skills status --short
git -C ~/hairpin_ws/src/Universal_Robots_ROS2_Driver rev-parse HEAD
git -C ~/hairpin_ws/src/Universal_Robots_ROS2_Driver diff --check
ros2 doctor --report
```

Ein schmutziger Working Tree ist kein ausreichender Versionsnachweis. Alle für
die Arbeit verwendeten Nodes, Konfigurationen, Launch-Dateien und Patches
müssen vor der Übergabe committed und mit einem eindeutigen Git-Tag versehen
werden.
