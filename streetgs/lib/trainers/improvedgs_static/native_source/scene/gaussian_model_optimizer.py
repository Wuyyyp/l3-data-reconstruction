"""
Optimizer and checkpoint helpers for GaussianModel.

The Gaussian model changes tensor sizes during densification and pruning. These
helpers build optimizer parameter groups, restore matching optimizer state from
checkpoints, and replace optimizer-owned tensors whenever the model structure
changes.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from utils.general_utils import get_expon_lr_func

try:
    from diff_gaussian_rasterization import GaussianAdam
except Exception:
    GaussianAdam = None


class GaussianModelOptimizerMixin:
    """
        Mixin that owns optimizer setup, restore, and tensor replacement logic.
    """

    def _ensure_optimizer_parameter_layouts(self) -> None:
        """
            Make optimizer-updated parameters contiguous before optimizer creation.
        """
        parameter_names = [
            "_xyz",
            "_features_dc",
            "_features_rest",
            "_opacity",
            "_scaling",
            "_rotation",
        ]
        for parameter_name in parameter_names:
            parameter = getattr(self, parameter_name)
            if not isinstance(parameter, nn.Parameter):
                continue
            if parameter.is_contiguous():
                continue
            setattr(
                self,
                parameter_name,
                nn.Parameter(parameter.detach().contiguous().requires_grad_(parameter.requires_grad)),
            )

    def _get_optimizer_step_value(self, stored_state: dict[str, object] | None) -> int:
        """
            Read an optimizer step counter whether it is stored as a tensor or int.
        """
        if stored_state is None:
            return 0
        step_value = stored_state.get("step")
        if step_value is None:
            return 0
        if torch.is_tensor(step_value):
            return int(step_value.item())
        return int(step_value)

    def _build_optimizer_param_groups(self, training_args: Any) -> list[dict[str, object]]:
        """
            Build one optimizer parameter group per Gaussian tensor family.

            Separate groups keep learning rates independent for xyz, color,
            opacity, scaling, and rotation.
        """
        lr_rate = float(getattr(training_args, "lr_rate", 1.0))
        param_groups = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale * lr_rate, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr * lr_rate, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr * lr_rate / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr * lr_rate, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr * lr_rate, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr * lr_rate, "name": "rotation"}
        ]
        return param_groups

    def _configure_optimizers(self, training_args: Any) -> None:
        """
            Recreate optimizers and learning-rate schedules from current tensors.

            This is called both at initial setup and after methods such as MiniGS
            reinitialization replace the whole Gaussian set.
        """
        self._ensure_optimizer_parameter_layouts()
        param_groups = self._build_optimizer_param_groups(training_args)
        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "ours_adam":
            if GaussianAdam is None:
                raise RuntimeError("GaussianAdam is not available, please install the correct rasterizer build.")
            self.optimizer = GaussianAdam(param_groups, lr=0.0, eps=1e-15, max_steps=int(getattr(training_args, "iterations", 30000)))
        else:
            raise ValueError("Unsupported optimizer_type: {}".format(self.optimizer_type))

        self.exposure_optimizer = torch.optim.Adam([self._exposure])
        lr_rate = float(getattr(training_args, "lr_rate", 1.0))
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale * lr_rate,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale * lr_rate,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )
        self.exposure_scheduler_args = get_expon_lr_func(
            training_args.exposure_lr_init,
            training_args.exposure_lr_final,
            lr_delay_steps=training_args.exposure_lr_delay_steps,
            lr_delay_mult=training_args.exposure_lr_delay_mult,
            max_steps=training_args.iterations,
        )

    def _restore_optimizer_state(self, opt_dict: dict[str, object]) -> None:
        """
            Restore checkpoint optimizer state by parameter group name.

            Tensor states are copied only when their shape still matches the
            current parameter, which avoids invalid momentum after SH expansion
            or other structure changes.
        """
        current_state = self.optimizer.state_dict()
        checkpoint_groups = opt_dict.get("param_groups", [])
        checkpoint_state = opt_dict.get("state", {})
        checkpoint_groups_by_name = {group.get("name"): group for group in checkpoint_groups}
        current_tensor_by_name = {
            "xyz": self._xyz,
            "f_dc": self._features_dc,
            "f_rest": self._features_rest,
            "opacity": self._opacity,
            "scaling": self._scaling,
            "rotation": self._rotation,
        }
        restored_state: dict[int, dict[str, torch.Tensor]] = {}
        for current_group in current_state["param_groups"]:
            group_name = current_group.get("name")
            checkpoint_group = checkpoint_groups_by_name.get(group_name)
            if checkpoint_group is None:
                continue
            for field_name, field_value in checkpoint_group.items():
                if field_name == "params":
                    continue
                current_group[field_name] = field_value
            checkpoint_param_id = checkpoint_group["params"][0]
            current_param_id = current_group["params"][0]
            state_entry = checkpoint_state.get(checkpoint_param_id)
            if state_entry is None:
                continue
            exp_avg = state_entry.get("exp_avg")
            if exp_avg is not None and tuple(exp_avg.shape) != tuple(current_tensor_by_name[group_name].shape):
                continue
            normalized_state_entry: dict[str, torch.Tensor | object] = {}
            for state_name, state_value in state_entry.items():
                if not torch.is_tensor(state_value):
                    normalized_state_entry[state_name] = state_value
                    continue
                normalized_state_entry[state_name] = state_value.contiguous()
            restored_state[current_param_id] = normalized_state_entry

        current_state["state"] = restored_state
        self.optimizer.load_state_dict(current_state)

    def restore(self, model_args: tuple[Any, ...], training_args: Any) -> None:
        """
            Restore model tensors, gradient accumulators, and optimizer state.
        """
        if len(model_args) != 15:
            raise ValueError("Only the 15-field 3DGS checkpoint format is supported; got {} fields.".format(len(model_args)))
        (
            self.active_sh_degree,
            self.max_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            _discarded_aux,
            xyz_gradient_accum,
            xyz_gradient_accum_abs,
            denom,
            opt_dict,
            self.spatial_lr_scale,
            _discarded_ids,
        ) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.xyz_gradient_accum_abs = xyz_gradient_accum_abs
        self.denom = denom
        self._restore_optimizer_state(opt_dict)

    def training_setup(self, training_args: Any) -> None:
        """
            Initialize accumulators and optimizers for the current Gaussian tensors.
        """
        self.percent_dense = training_args.percent_dense
        self.noise_lr = float(getattr(training_args, "noise_lr", 0.0))
        self.mcmc_noise_opacity_sharpness = float(getattr(training_args, "mcmc_noise_opacity_sharpness", 100.0))
        if self.mcmc_noise_opacity_sharpness <= 0.0:
            raise ValueError("mcmc_noise_opacity_sharpness must be > 0.")
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        self._configure_optimizers(training_args)

    def update_learning_rate(self, iteration: int) -> float | None:
        """
            Update exposure and xyz learning rates for the current iteration.
        """
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def replace_tensor_to_optimizer(self, tensor: torch.Tensor, name: str) -> dict[str, nn.Parameter]:
        """
            Replace one optimizer parameter tensor and reset its momentum buffers.
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                tensor = tensor.contiguous()
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                    del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                if stored_state is not None:
                    self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def replace_tensors_to_optimizer(self, inds: torch.Tensor | None = None) -> dict[str, nn.Parameter]:
        """
            Rebind all optimizer parameter objects to current model tensors.

            When indices are provided, only those momentum entries are reset.
            This keeps useful optimizer history for unchanged Gaussians.
        """
        tensors_dict = {
            "xyz": self._xyz,
            "f_dc": self._features_dc,
            "f_rest": self._features_rest,
            "opacity": self._opacity,
            "scaling": self._scaling,
            "rotation": self._rotation,
        }
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            tensor = tensors_dict[group["name"]].contiguous()
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                if inds is not None:
                    stored_state["exp_avg"][inds] = 0
                    stored_state["exp_avg_sq"][inds] = 0
                else:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                del self.optimizer.state[group["params"][0]]
            group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
            if stored_state is not None:
                self.optimizer.state[group["params"][0]] = stored_state
            optimizable_tensors[group["name"]] = group["params"][0]

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        torch.cuda.empty_cache()
        return optimizable_tensors

    def _prune_optimizer(self, mask: torch.Tensor) -> dict[str, nn.Parameter]:
        """
            Apply a keep mask to all optimizer parameters and matching state tensors.
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask].contiguous()
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask].contiguous()
                if "last_seen_step" in stored_state and torch.is_tensor(stored_state["last_seen_step"]):
                    stored_state["last_seen_step"] = stored_state["last_seen_step"][mask].contiguous()
                if "lr_history" in stored_state and torch.is_tensor(stored_state["lr_history"]):
                    stored_state["lr_history"] = stored_state["lr_history"].contiguous()

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(group["params"][0][mask].contiguous().requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].contiguous().requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def cat_tensors_to_optimizer(self, tensors_dict: dict[str, torch.Tensor]) -> dict[str, nn.Parameter]:
        """
            Append new Gaussian tensors to optimizer parameters and extend state buffers.
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]].contiguous()
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0).contiguous()
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0).contiguous()
                if "last_seen_step" in stored_state and torch.is_tensor(stored_state["last_seen_step"]):
                    append_step = torch.full(
                        (extension_tensor.shape[0],),
                        self._get_optimizer_step_value(stored_state),
                        dtype=stored_state["last_seen_step"].dtype,
                        device=extension_tensor.device,
                    )
                    stored_state["last_seen_step"] = torch.cat((stored_state["last_seen_step"], append_step), dim=0).contiguous()
                if "lr_history" in stored_state and torch.is_tensor(stored_state["lr_history"]):
                    stored_state["lr_history"] = stored_state["lr_history"].contiguous()

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).contiguous().requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).contiguous().requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def update_opacity_lr(self, rate: float) -> None:
        """
            Let GNS raise the opacity learning rate during regularized pruning and restore it afterward.
        """
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "opacity":
                param_group["lr"] = param_group["lr"] * float(rate)
