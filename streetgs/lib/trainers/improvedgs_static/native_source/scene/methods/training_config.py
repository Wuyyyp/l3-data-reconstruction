"""
Training-method configuration parser.

The command line exposes several method names and component switches. This
module normalizes the method name, validates incompatible settings early, and
returns a compact config dictionary used by all training stages.
"""
from typing import Any

VALID_METHODS = {"3dgs", "absgs", "minigs", "mcmc", "improvedgs", "gns"}


def _read_bool(opt: Any, field_name: str, default_value: bool) -> bool:
    """
        Read a boolean option and reject non-boolean values after argparse parsing.
    """
    raw_value = getattr(opt, field_name, default_value)
    if isinstance(raw_value, bool):
        return raw_value
    raise ValueError("{} only accepts true/false.".format(field_name))


def build_training_method_config(opt: Any) -> dict[str, Any]:
    """
        Normalize the selected training method and resolve method component flags.
    """
    method = str(getattr(opt, "training_method", "3dgs")).strip().lower()
    if method == "default":
        method = "3dgs"
    if method not in VALID_METHODS:
        raise ValueError("Unsupported training_method: {}".format(method))
    if method == "mcmc" and int(getattr(opt, "budget", 0)) <= 0:
        raise ValueError("mcmc requires budget > 0.")

    if method in ("improvedgs", "gns"):
        use_las = _read_bool(opt, "use_las", True)
        use_eas = _read_bool(opt, "use_eas", True)
        use_mu = _read_bool(opt, "use_mu", True)
        use_rap = _read_bool(opt, "use_rap", True) if method == "improvedgs" else True
    else:
        use_las = False
        use_eas = False
        use_mu = False
        use_rap = False

    return {
        "training_method": method,
        "use_las": use_las,
        "use_eas": use_eas,
        "use_mu": use_mu,
        "use_rap": use_rap,
    }
