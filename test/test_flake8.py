# Copyright 2017 Open Source Robotics Foundation, Inc.
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

from ament_flake8.main import main_with_errors
from pathlib import Path
import pytest


@pytest.mark.flake8
@pytest.mark.linter
def test_flake8():
    root = Path(__file__).parents[1]
    paths = [
        root / 'hairpin_skills' / 'ati_rdt_wrench.py',
        root / 'hairpin_skills' / 'phase_b_spi_data_collection.py',
        root / 'hairpin_skills' / 'radial_sliding_test.py',
        root / 'launch' / 'ur3e_bringup.launch.py',
        root / 'setup.py',
        root / 'tools' / 'validate_run_manifest.py',
    ]
    config = root / 'test' / 'flake8.ini'
    rc, errors = main_with_errors(
        argv=['--config', str(config), *(str(path) for path in paths)]
    )
    assert rc == 0, \
        'Found %d code style errors / warnings:\n' % len(errors) + \
        '\n'.join(errors)
