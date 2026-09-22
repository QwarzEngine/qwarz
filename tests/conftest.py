"""Collection policy for environment-dependent test modules.

Runtime tests import ``torch`` (installed in the donor virtualenv, not in the
uv environment) and campaign companion tests import from their local
``results/<campaign>/`` modules, which are gitignored and may drift after a
campaign closes. Skip the modules that cannot import in the current
environment instead of failing the whole collection.
"""

import importlib
import sys

ENVIRONMENT_DEPENDENT = [
    # torch lives in the donor venv
    "test_runtime_engine",
    "test_runtime_worker",
    "test_runtime_vision",
    "test_runtime_rendering",
    "test_runtime_startup",
    "test_tool_diagnostics",
    # campaign companions under results/
    "test_xqa_fase3",
    "test_gdn_native_geometry",
    "test_walker_overhead",
    "test_attention_cubin_v2",
    "test_engine_topology_geometry",
    "test_engine_topology_superblocks",
]

collect_ignore = []

for _name in ENVIRONMENT_DEPENDENT:
    if f"tests.{_name}" in sys.modules:
        continue
    try:
        importlib.import_module(f"tests.{_name}")
    except BaseException:
        collect_ignore.append(f"{_name}.py")
