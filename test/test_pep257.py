# Copyright 2015 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ament_pep257.main import main
from pathlib import Path
import pytest


@pytest.mark.linter
@pytest.mark.pep257
def test_pep257():
    root = Path(__file__).parents[1]
    paths = [
        root / 'hairpin_skills' / 'ati_rdt_wrench.py',
        root / 'hairpin_skills' / 'phase_b_spi_data_collection.py',
        root / 'hairpin_skills' / 'radial_sliding_test.py',
        root / 'launch' / 'ur3e_bringup.launch.py',
        root / 'setup.py',
        root / 'tools' / 'validate_run_manifest.py',
    ]
    rc = main(argv=[str(path) for path in paths])
    assert rc == 0, 'Found code style errors / warnings'
