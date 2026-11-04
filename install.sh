#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  echo "Usage: bash install.sh {all|pose|dvp|mesh|streetgs} [path/to/base-python]"; exit 0
fi
if [[ "${1:-}" == all ]]; then
  if (( $# != 1 )); then echo "For custom base interpreters install each component separately." >&2; exit 2; fi
  for item in pose dvp mesh streetgs; do bash "$root/install.sh" "$item"; done
  exit 0
fi
case "${1:-}" in
  pose) dir=pose-repair; version=3.11; torch=2.8.0; vision=0.23.0; cuda=cu128 ;;
  dvp) dir=dvp-openmvs; version=3.11; torch=2.8.0; vision=0.23.0; cuda=cu128 ;;
  mesh) dir=vehicle-mesh; version=3.12; torch=2.7.1; vision=0.22.1; cuda=cu128 ;;
  streetgs) dir=streetgs; version=3.10; torch=2.1.1; vision=0.16.1; cuda=cu118 ;;
  *) echo "Usage: bash install.sh {all|pose|dvp|mesh|streetgs} [path/to/base-python]" >&2; exit 2 ;;
esac
component=$1
python=${2:-python$version}
venv="$root/.envs/$component"
require_version() {
  "$1" -c 'import sys; actual="%d.%d" % sys.version_info[:2]; sys.exit(0 if actual==sys.argv[1] else f"Expected Python {sys.argv[1]}, got {actual}")' "$version"
}
if [[ ! -x "$venv/bin/python" ]]; then
  if [[ -e "$venv" ]]; then echo "Incomplete environment: $venv" >&2; exit 1; fi
  require_version "$python"
  "$python" -m venv "$venv"
fi
py="$venv/bin/python"; require_version "$py"
cd "$root/$dir"
"$py" -m pip install --upgrade pip
"$py" -m pip install -r "$root/requirements.txt"
"$py" -m pip install --index-url "https://download.pytorch.org/whl/$cuda" "torch==$torch+$cuda" "torchvision==$vision+$cuda"
# Build dependencies of CUDA extensions include torch from this venv.
if [[ "$component" == dvp ]]; then
  "$py" -m pip install "numpy==2.2.6"
  "$py" -m pip install --no-build-isolation -r requirements.txt
else
  "$py" -m pip install -r requirements.txt
fi
if [[ "$component" == streetgs ]]; then
  "$py" -m pip install --no-build-isolation -r requirements-native.txt
fi
"$py" -m pip check
echo "Python dependencies installed for $component. See INSTALL.md for native tools and weights."
