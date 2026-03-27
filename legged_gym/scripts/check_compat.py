import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
RSL_RL_ROOT = os.path.join(REPO_ROOT, "rsl_rl")

for path in (REPO_ROOT, RSL_RL_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from legged_gym.utils import collect_runtime_report


def main() -> int:
    report = collect_runtime_report()
    print(json.dumps(report, indent=2))
    return 1 if report["warnings"] else 0


if __name__ == "__main__":
    sys.exit(main())
