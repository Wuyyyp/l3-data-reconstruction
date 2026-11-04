"""
Small stages called from the main training loop.

These helpers keep the loop in `train.py` readable. They update progress output,
run scheduled evaluation and saving, adjust learning-rate/SH schedules, and
perform optimizer steps according to the active training method.
"""
from __future__ import annotations

import os
from typing import Any

import torch

from gaussian_renderer import render
from scene.gaussian_model import GaussianModel as GaussianModel3DGS
from utils.loss_utils import l1_loss
from scene.training_context import TrainingContext
from scene.training_runtime import TrainingLoopState, save_training_checkpoint, synchronized_timestamp, training_report


def _resolve_dense_optimizer_type(opt: Any, model_label: str = "3DGS") -> str:
    """
        Validate and return the configured optimizer type for dense Gaussian tensors.
    """
    configured_optimizer_type = opt.optimizer_type
    if configured_optimizer_type not in {"default", "ours_adam"}:
        raise ValueError(
            "{} currently supports only optimizer_type in {{default, ours_adam}}; got {}.".format(
                str(model_label).upper(),
                configured_optimizer_type,
            )
        )
    return configured_optimizer_type


def update_training_progress_bar(
    progress_bar: Any,
    iteration: int,
    ema_loss_for_log: float,
    loss_value: float,
    num_gaussians: int,
) -> float:
    """
        Update the tqdm postfix with smoothed loss and current Gaussian count.
    """
    ema_loss_for_log = 0.4 * loss_value + 0.6 * ema_loss_for_log
    if iteration % 10 == 0:
        progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.4f}", "N_GS": f"{num_gaussians}"})
        progress_bar.update(10)
    return ema_loss_for_log


def run_log_and_save_stage(
    context: TrainingContext,
    gaussians: GaussianModel3DGS,
    loop_state: TrainingLoopState,
    iteration: int,
) -> float:
    """
        Run scheduled evaluation, model saving, and checkpoint saving.

        The return value is the time spent in reporting, which the caller
        subtracts from pure training time.
    """
    scene = context.scene
    if scene is None:
        raise ValueError("Training context is missing scene during log/save stage.")

    runtime_args = context.runtime_args
    dataset = context.dataset
    pipe = context.pipe
    report_time_seconds = 0.0
    if iteration in loop_state["testing_iterations"]:
        report_start = synchronized_timestamp()
        training_report(
            iteration,
            l1_loss,
            scene,
            render,
            (pipe, loop_state["background"], 1.0, dataset.train_test_exp, False, False, None),
            dataset.train_test_exp,
            loop_state["report_lpips_model"],
            bool(getattr(runtime_args, "report_lpips_test", False)),
            bool(getattr(runtime_args, "report_lpips_train", False)),
            loop_state["log_path"],
            iteration == min(loop_state["testing_iterations"]),
            context.runtime_state,
        )
        report_time_seconds += synchronized_timestamp() - report_start
    if iteration in loop_state["saving_iterations"]:
        print("\n[ITER {}] Saving Gaussians".format(iteration))
        scene.save(iteration)
        bilateral_grid = context.runtime_state.get("bilateral_grid")
        if bilateral_grid is not None:
            torch.save(bilateral_grid.capture(), os.path.join(scene.model_path, "bilateral_grid{}.pth".format(iteration)))
    if iteration in loop_state["checkpoint_iterations"]:
        print("\n[ITER {}] Saving Checkpoint".format(iteration))
        save_training_checkpoint(
            os.path.join(scene.model_path, "chkpnt{}.pth".format(iteration)),
            gaussians,
            iteration,
            context.runtime_state,
        )
    return report_time_seconds


def _apply_minigs_sh_schedule(context: TrainingContext, gaussians: GaussianModel3DGS, iteration: int) -> bool:
    """
        Apply MiniGS's delayed SH-degree schedule.

        MiniGS keeps low SH capacity early, then restores the dataset target
        degree and resumes normal SH increments after the restore iteration.
    """
    if str(context.method_config.get("training_method", "")).lower() != "minigs":
        return False

    sh_restore_iter = 11_000
    if int(iteration) < sh_restore_iter:
        return True
    target_sh_degree = int(context.runtime_state.get("dataset_sh_degree") or 3)
    if int(getattr(gaussians, "max_sh_degree", 0)) < target_sh_degree:
        gaussians.ensure_sh_degree_capacity(target_sh_degree)
    if iteration % 1000 == 0 and iteration > sh_restore_iter:
        gaussians.oneupSHdegree()
    return True


def _validate_dense_optimizer_type(
    context: TrainingContext,
    gaussians: GaussianModel3DGS,
    model_label: str = "3DGS",
) -> None:
    """
        Ensure the model optimizer type matches the current config.
    """
    configured_optimizer_type = _resolve_dense_optimizer_type(context.opt, model_label)
    if str(getattr(gaussians, "optimizer_type", "default")).lower() != configured_optimizer_type:
        raise ValueError(
            "GaussianModel{} optimizer_type={} does not match current config {}.".format(
                str(model_label).upper(),
                getattr(gaussians, "optimizer_type", "default"),
                configured_optimizer_type,
            )
        )


def update_3dgs_training_schedule(context: TrainingContext, gaussians: GaussianModel3DGS, iteration: int) -> None:
    """
        Update learning rate and SH degree before rendering the current iteration.
    """
    _validate_dense_optimizer_type(context, gaussians, "3dgs")
    gaussians.update_learning_rate(int(iteration))
    if not _apply_minigs_sh_schedule(context, gaussians, iteration) and iteration % 1000 == 0:
        gaussians.oneupSHdegree()


def should_run_parameter_update(opt: Any, iteration: int, method: str, use_mu: bool) -> bool:
    """
        Decide whether this iteration should step trainable parameters.

        ImprovedGS/GNS can use MU, which skips dense optimizer steps later in
        training to reduce update frequency. Other methods step every iteration
        until the final one.
    """
    if iteration >= int(opt.iterations):
        return False
    if method not in ("improvedgs", "gns") or not use_mu:
        return True

    mu_start_iter = int(opt.mu_start_iter)
    if mu_start_iter < 0:
        mu_start_iter = 20_000
    mu_interval = max(int(opt.mu_interval), 1)
    if int(opt.mu_interval) <= 1:
        mu_interval = 5

    mu_second_start_iter = int(opt.mu_second_start_iter)
    mu_second_interval = max(int(opt.mu_second_interval), 1)
    if mu_second_start_iter <= mu_start_iter:
        mu_second_start_iter = mu_start_iter + 1

    return (
        iteration < mu_start_iter
        or (mu_start_iter <= iteration < mu_second_start_iter and iteration % mu_interval == 0)
        or (iteration >= mu_second_start_iter and iteration % mu_second_interval == 0)
    )


def _step_dense_optimizer(
    gaussians: GaussianModel3DGS,
    optimizer_type: str,
    step_size: int,
    iteration: int,
) -> None:
    """
        Step either the custom GaussianAdam optimizer or the default Adam optimizer.
    """
    if optimizer_type == "ours_adam":
        gaussians.optimizer.step(step_size, iteration)
    else:
        gaussians.optimizer.step()
    gaussians.optimizer.zero_grad(set_to_none=True)


def run_3dgs_parameter_update_method(
    context: TrainingContext,
    gaussians: GaussianModel3DGS,
    iteration: int,
    radii: torch.Tensor,
) -> None:
    """
        Apply optimizer updates for one iteration.

        Exposure is always stepped with its own Adam optimizer. Dense Gaussian
        tensors follow the method schedule; MCMC also injects position noise
        after the xyz update.
    """
    method = str(context.method_config["training_method"]).lower()
    use_mu = bool(context.method_config.get("use_mu", False))
    bilateral_grid = context.runtime_state.get("bilateral_grid")
    if bilateral_grid is not None:
        if iteration < int(context.opt.iterations):
            bilateral_grid.optimizer.step()
            if bilateral_grid.scheduler is not None:
                bilateral_grid.scheduler.step()
        bilateral_grid.optimizer.zero_grad(set_to_none=True)
    if not should_run_parameter_update(context.opt, iteration, method, use_mu):
        return

    optimizer_type = _resolve_dense_optimizer_type(context.opt, "3dgs")
    gaussians.exposure_optimizer.step()
    gaussians.exposure_optimizer.zero_grad(set_to_none=True)
    _step_dense_optimizer(gaussians, optimizer_type, radii.shape[0], iteration)

    if method == "mcmc":
        for param_group in gaussians.optimizer.param_groups:
            if param_group["name"] == "xyz":
                gaussians.apply_mcmc_noise(float(param_group["lr"]))
                return
        raise ValueError("Failed to resolve xyz learning rate from optimizer.")
