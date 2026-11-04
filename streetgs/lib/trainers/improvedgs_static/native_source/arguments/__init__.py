"""
Command-line argument system.

Each parameter group declares defaults as class attributes. ParamGroup then
turns those attributes into argparse options and later extracts only the fields
that belong to that group. This keeps the training entry point readable while
still producing normal Python objects for the rest of the code.
"""
from __future__ import annotations
import ast
from argparse import ArgumentParser, Namespace
import os
import sys


def parse_bool_arg(raw_value: str | bool) -> bool:
    """
        Parse flexible CLI boolean text into real `True` or `False` values.

        Argparse passes strings for values such as `--flag false`; this helper
        accepts common spellings and raises a clear error for anything else.
    """
    if isinstance(raw_value, bool):
        return raw_value
    normalized = str(raw_value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError("Unsupported boolean value: {}".format(raw_value))


class GroupParams:
    """
        Simple object used to hold extracted parameters as attributes.

        The project historically reads options with dot access, so this class
        intentionally stays empty and lets ParamGroup attach fields dynamically.
    """
    pass


class ParamGroup:
    """
        Base class that registers one group of argparse options from defaults.
    """
    def __init__(self, parser: ArgumentParser, name: str, fill_none: bool = False) -> None:
        self._group_name = name
        self._fill_none = fill_none
        if not hasattr(self, "_fields"):
            self._fields: dict[str, object] = {}
        self._register_group(parser)

    def _register_group(self, parser: ArgumentParser) -> None:
        """
            Convert fields in `self._fields` into argparse options.

            A leading underscore means the option also gets a one-letter short
            form, for example `_source_path` becomes `--source_path` and `-s`.
        """
        group = parser.add_argument_group(self._group_name)
        for raw_key, raw_value in self._fields.items():
            shorthand = False
            key = raw_key
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            value = raw_value if not self._fill_none else None
            value_type = type(raw_value)
            if shorthand:
                if value_type is bool or raw_value is None:
                    group.add_argument(
                        "--" + key,
                        "-" + key[0:1],
                        default=value,
                        nargs="?",
                        const=True,
                        type=parse_bool_arg,
                    )
                else:
                    group.add_argument("--" + key, "-" + key[0:1], default=value, type=value_type)
            else:
                if value_type is bool or raw_value is None:
                    group.add_argument(
                        "--" + key,
                        default=value,
                        nargs="?",
                        const=True,
                        type=parse_bool_arg,
                    )
                else:
                    group.add_argument("--" + key, default=value, type=value_type)

    def extract(self, args: Namespace) -> GroupParams:
        """
            Copy only this group's parsed fields from the shared argparse namespace.
        """
        group = GroupParams()
        for raw_key, raw_default in self._fields.items():
            key = raw_key[1:] if raw_key.startswith("_") else raw_key
            value = getattr(args, key, raw_default)
            if self._fill_none and value is None:
                value = raw_default
            setattr(group, key, value)
        return group


class ModelParams(ParamGroup):
    """
        Dataset and output-path options used before the scene is created.
    """
    def __init__(self, parser: ArgumentParser, sentinel: bool = False) -> None:
        self.sh_degree = 3  # Maximum SH degree
        self._source_path = ""  # Dataset path
        self._model_path = ""  # Output model path
        self._images = "images"  # Image subdirectory
        self._depths = ""  # Depth-map subdirectory
        self._resolution = -1  # Training resolution scale
        self._white_background = False  # Use a white background
        self.init_type = "sfm"  # Initialization type
        self.train_test_exp = False  # Enable train/test exposure
        self.data_device = "cuda"  # Image data loading device
        self.eval = True  # Use evaluation split
        fields = {key: value for key, value in vars(self).items() if not key.startswith("__")}
        self._fields = fields
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args: Namespace) -> GroupParams:
        """
            Extract model options and normalize the source path to an absolute path.
        """
        group = super().extract(args)
        group.source_path = os.path.abspath(group.source_path)
        return group


class PipelineParams(ParamGroup):
    """
        Renderer behavior options passed to the Gaussian rasterizer.
    """
    def __init__(self, parser: ArgumentParser, sentinel: bool = False) -> None:
        self.debug = False  # Renderer debug switch
        self.antialiasing = False  # Enable antialiasing
        self.depth_ratio = 0.0  # Depth blending ratio
        self._fields = {key: value for key, value in vars(self).items() if not key.startswith("__")}
        super().__init__(parser, "Pipeline Parameters", sentinel)


class OptimizationBaseParams(ParamGroup):
    """
        Common optimization settings shared by all training methods.
    """
    def __init__(self, parser: ArgumentParser) -> None:
        self.iterations = 30_000  # Total training iterations
        self.position_lr_init = 0.00004  # Initial position learning rate
        self.position_lr_final = 0.000002  # Final position learning rate
        self.position_lr_delay_mult = 0.01  # Position LR delay multiplier
        self.position_lr_max_steps = 30_000  # Position LR decay steps
        self.lr_rate = 1.0  # Global LR multiplier
        self.feature_lr = 0.0025  # SH feature learning rate
        self.opacity_lr = 0.025  # Opacity learning rate
        self.initial_opacity = 0.1  # Initial opacity
        self.scaling_lr = 0.005  # Scale learning rate
        self.rotation_lr = 0.001  # Rotation learning rate
        self.exposure_lr_init = 0.01  # Initial exposure learning rate
        self.exposure_lr_final = 0.001  # Final exposure learning rate
        self.exposure_lr_delay_steps = 0  # Exposure delay steps
        self.exposure_lr_delay_mult = 0.0  # Exposure delay multiplier
        self.percent_dense = 0.01  # Clone/split scale boundary
        self.lambda_dssim = 0.2  # DSSIM loss weight
        self.use_bilateral_grid = False  # Correct only the L1 reconstruction branch
        self.bilateral_grid_x = 16
        self.bilateral_grid_y = 16
        self.bilateral_grid_w = 8
        self.bilateral_grid_lr = 0.002
        self.bilateral_grid_lr_schedule = True
        self.bilateral_grid_adam_eps = 1e-15
        self.bilateral_grid_tv_weight = 10.0
        self.bilateral_grid_detached_training = False
        self.bilateral_grid_sample_fraction = 1.0
        self.densification_interval = 100  # Densification interval
        self.opacity_reset_interval = 3000  # Opacity reset interval
        self.densify_from_iter = 500  # Densification start iteration
        self.densify_until_iter = 15_000  # Densification end iteration
        self.densify_grad_threshold = 0.0003  # Densification gradient threshold
        self.min_opacity = 0.005  # Minimum opacity pruning threshold
        self.depth_l1_weight_init = 1.0  # Initial depth regularization weight
        self.depth_l1_weight_final = 0.01  # Final depth regularization weight
        self.random_background = False  # Use random background
        self.optimizer_type = "ours_adam"  # Dense optimizer type
        self._fields = {key: value for key, value in vars(self).items() if not key.startswith("__")}
        super().__init__(parser, "Optimization Base Parameters")


class TrainingMethodParams(ParamGroup):
    """
        Method selector and ImprovedGS component switches.
    """
    def __init__(self, parser: ArgumentParser) -> None:
        self.training_method = "improvedgs"  # Overall training strategy
        self.use_las = True  # Enable ImprovedGS long-axis split
        self.use_rap = True  # Enable ImprovedGS RAP
        self.use_eas = True  # Enable ImprovedGS edge importance
        self.use_mu = True  # Enable ImprovedGS MU update cadence
        self._fields = {key: value for key, value in vars(self).items() if not key.startswith("__")}
        super().__init__(parser, "Training Method Parameters")


class MiniGSParams(ParamGroup):
    """
        MiniGS-specific limits and periodic reinitialization settings.
    """
    def __init__(self, parser: ArgumentParser) -> None:
        self.minigs_imp_metric = "outdoor"  # MiniGS importance metric
        self.minigs_num_depth = 3_500_000  # Depth back-sampling sample count
        self.minigs_num_max = 4_500_000  # MiniGS maximum Gaussian count
        self.minigs_reinit_interval = 5_000  # MiniGS reinitialization interval
        self.minigs_blur_screen_coverage_divisor = 5_000.0  # Blur area threshold divisor
        self._fields = {key: value for key, value in vars(self).items() if not key.startswith("__")}
        super().__init__(parser, "MiniGS Parameters")


class ImprovedGSParams(ParamGroup):
    """
        ImprovedGS structure-growth, budget, RAP, and MU schedule options.
    """
    def __init__(self, parser: ArgumentParser) -> None:
        self.edge_sample_cams = 10  # Edge-scoring sampled camera count
        self.improvedgs_reset_min_opacity = 0.05  # RAP reset opacity upper bound
        self.split_distance = 0.45  # Long-axis split offset ratio
        self.opacity_reduction = 0.6  # Opacity decay after split
        self.budget = 300_0000  # Gaussian budget
        self.budget_multiplier = 3.0  # Budget warmup multiplier
        self.budget_warmup_until_offset = 500  # Budget warmup end offset
        self.rap_prune_ratio = 0.2  # RAP pruning ratio
        self.rap_prune_offset = 300  # Pruning delay after RAP reset
        self.rap_rounds = 2  # Maximum RAP rounds
        self.mu_start_iter = 20_000  # MU first-stage start iteration
        self.mu_interval = 5  # MU first-stage step interval
        self.mu_second_start_iter = 24_000  # MU second-stage start iteration
        self.mu_second_interval = 20  # MU second-stage step interval
        self._fields = {key: value for key, value in vars(self).items() if not key.startswith("__")}
        super().__init__(parser, "ImprovedGS Parameters")


class MCMCParams(ParamGroup):
    """
        MCMC relocation and noise regularization settings.
    """
    def __init__(self, parser: ArgumentParser) -> None:
        self.noise_lr = 5e5  # MCMC noise strength coefficient
        self.mcmc_noise_opacity_sharpness = 100.0  # MCMC opacity noise gate sharpness
        self.mcmc_scale_reg = 0.01  # MCMC scale regularization weight
        self.mcmc_opacity_reg = 0.01  # MCMC opacity regularization weight
        self._fields = {key: value for key, value in vars(self).items() if not key.startswith("__")}
        super().__init__(parser, "MCMC Parameters")


class GNSParams(ParamGroup):
    """
        GNS opacity regularization and learning-rate scaling settings.
    """
    def __init__(self, parser: ArgumentParser) -> None:
        self.gns_opacity_reg = 0.002  # GNS opacity regularization weight
        self.gns_opacity_lr_scale = 5.0  # GNS opacity learning-rate multiplier
        self._fields = {key: value for key, value in vars(self).items() if not key.startswith("__")}
        super().__init__(parser, "GNS Parameters")


class RegPruneParams(ParamGroup):
    """
        Final pruning budget settings used by regularized pruning methods.
    """
    def __init__(self, parser: ArgumentParser) -> None:
        self.final_budget = 60_0000  # Target Gaussian count for pruning
        self.final_budget_mode = "rate"  # Target budget mode
        self.final_rate = 0.2  # Retention ratio in rate mode
        self.reg_prune_from_iter = 15_000  # Regularized pruning start iteration
        self.reg_prune_until_iter = 20_000  # Regularized pruning end iteration
        self._fields = {key: value for key, value in vars(self).items() if not key.startswith("__")}
        super().__init__(parser, "Regularized Pruning Parameters")


class OptimizationParams:
    """
        Wrapper that registers and merges all optimization-related groups.
    """
    def __init__(self, parser: ArgumentParser) -> None:
        self.groups = [
            OptimizationBaseParams(parser),
            TrainingMethodParams(parser),
            MiniGSParams(parser),
            ImprovedGSParams(parser),
            MCMCParams(parser),
            GNSParams(parser),
            RegPruneParams(parser),
        ]

    def extract(self, args: Namespace) -> GroupParams:
        """
            Merge all optimization parameter groups into one object for training.
        """
        merged = GroupParams()
        for group in self.groups:
            partial = group.extract(args)
            for key, value in vars(partial).items():
                setattr(merged, key, value)
        return merged


def parse_cfg_args_namespace(cfgfile_string: str) -> Namespace:
    """
        Safely parse the legacy `cfg_args` Namespace(...) text format.
    """
    cfgfile_string = str(cfgfile_string).strip()
    if cfgfile_string == "":
        return Namespace()
    expression = ast.parse(cfgfile_string, mode="eval").body
    if not isinstance(expression, ast.Call) or not isinstance(expression.func, ast.Name):
        raise ValueError("cfg_args must contain a Namespace(...) expression.")
    if expression.func.id != "Namespace":
        raise ValueError("cfg_args must use Namespace(...).")
    if expression.args:
        raise ValueError("cfg_args only supports keyword arguments.")

    parsed_values = {}
    for keyword in expression.keywords:
        if keyword.arg is None:
            raise ValueError("cfg_args does not support **kwargs.")
        parsed_values[keyword.arg] = ast.literal_eval(keyword.value)
    return Namespace(**parsed_values)


def get_combined_args(parser: ArgumentParser) -> Namespace:
    """
        Merge command-line arguments with the saved `cfg_args` file in model_path.

        Values provided on the command line take priority, while missing values
        can be recovered from the training run configuration saved with a model.
    """
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except (TypeError, FileNotFoundError, OSError):
        print("Config file not found at {}".format(getattr(args_cmdline, "model_path", "")))
    try:
        args_cfgfile = parse_cfg_args_namespace(cfgfile_string)
    except (SyntaxError, ValueError) as error:
        print("Could not parse cfg_args safely: {}".format(error))
        args_cfgfile = Namespace()

    merged_dict = vars(args_cfgfile).copy()
    for key, value in vars(args_cmdline).items():
        if value is not None:
            merged_dict[key] = value
    return Namespace(**merged_dict)
