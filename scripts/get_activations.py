from __future__ import annotations

import dataclasses
import logging
import pathlib
import pickle
from typing import Iterable

import jax
import numpy as np
import torch
import tyro
from tqdm import tqdm

from openpi.models import model as _model
from openpi.training import config as _config
from openpi.training import data_loader as _data


@dataclasses.dataclass
class Args:
    # 需要提取激活的两组权重路径
    finetuned_ckpt_dir: str = "/root/openpi/checkpoints/pi05_libero_base"
    base_ckpt_dir: str = "/root/openpi/checkpoints/pi05_base"

    # 使用的训练配置
    config_name: str = "pi05_libero"

    # 输出目录
    output_dir: str = "/root/openpi/libero_compare/"

    # 选择要抓取的层号
    activation_layers: list[int] = dataclasses.field(default_factory=lambda: [9])

    # 总共的 batch 数量
    num_batches: int = 273465

    # 每多少个样本聚合保存一次
    save_group_size: int = 100

    # batch 大小
    batch_size: int = 1

    # 是否跳过 norm stats（若本地没有 norm_stats.json，则设为 True）
    skip_norm_stats: bool = False

    # 设备选择
    device: str = "cuda"

    # 额外选项
    seed: int = 0
    num_workers: int = 0
    assets_base_dir: str = "/root/openpi/assets"
    
    # Steering 配置
    steering_method: str = "none" 
    steering_params: dict = dataclasses.field(default_factory=dict) 


def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def _prepare_config(args: Args) -> _config.TrainConfig:
    cfg = dataclasses.replace(_config.get_config(args.config_name))
    cfg = dataclasses.replace(
        cfg,
        exp_name="get_activations",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        wandb_enabled=False,
    )
    if args.assets_base_dir is not None:
        cfg = dataclasses.replace(cfg, assets_base_dir=args.assets_base_dir)
    
    if args.steering_method != "none":
        steering_config = {
            "method": args.steering_method,
            "params": args.steering_params,
        }
        cfg = dataclasses.replace(
            cfg,
            model=dataclasses.replace(cfg.model, steering_config=steering_config),
        )
    else:
        cfg = dataclasses.replace(
            cfg,
            model=dataclasses.replace(cfg.model, steering_config=None),
        )
    
    return cfg


def _create_loader(cfg: _config.TrainConfig, args: Args) -> _data.DataLoader:
    return _data.create_data_loader(
        cfg,
        framework="pytorch",
        shuffle=False,
        num_batches=args.num_batches,
        skip_norm_stats=args.skip_norm_stats,
    )


def _load_model(cfg: _config.TrainConfig, ckpt_dir: pathlib.Path, device: torch.device, layers: Iterable[int]):
    weight_path = ckpt_dir / "model.safetensors"
    if not weight_path.exists():
        raise FileNotFoundError(f"未找到模型权重: {weight_path}")

    model = cfg.model.load_pytorch(cfg, str(weight_path))
    model = model.to(device)
    model.eval()
    model.enable_activation_capture(list(layers))
    return model


def _observation_to_device(observation: _model.Observation, device: torch.device) -> _model.Observation:
    def _to_device(x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.to(device)
        return torch.as_tensor(np.asarray(x), device=device)

    obs_dict = observation.to_dict()
    obs_dict = jax.tree.map(_to_device, obs_dict)
    return _model.Observation.from_dict(obs_dict)


def _save_activations(
    save_root: pathlib.Path,
    tag: str,
    start_idx: int,
    end_idx: int,
    items: list[dict[str, object]],
):
    save_dir = save_root / tag
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"group_{start_idx:05d}_{end_idx:05d}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(items, f)
    logging.info("已保存激活: %s (包含 %d 条)", save_path, len(items))


def main(args: Args):
    _setup_logging()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    cfg = _prepare_config(args)

    loader = _create_loader(cfg, args)
    data_cfg = loader.data_config()

    output_dir = pathlib.Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ft_model = _load_model(cfg, pathlib.Path(args.finetuned_ckpt_dir), device, args.activation_layers)
    base_model = _load_model(cfg, pathlib.Path(args.base_ckpt_dir), device, args.activation_layers)
    models = {"finetuned": ft_model, "base": base_model}

    action_shape = (
        args.batch_size,
        cfg.model.action_horizon,
        cfg.model.action_dim,
    )

    buffers: dict[str, list[dict[str, object]]] = {"finetuned": [], "base": []}
    group_start: dict[str, int] = {"finetuned": 0, "base": 0}

    for batch_idx, (observation, actions) in enumerate(tqdm(loader, total=args.num_batches, desc="extract activations")):
        obs_on_device = _observation_to_device(observation, device)
        expert_actions = actions.cpu().numpy()
        prompt_tokens = (
            observation.tokenized_prompt.cpu().numpy() if observation.tokenized_prompt is not None else None
        )

        shared_noise = torch.randn(action_shape, device=device, dtype=torch.float32)

        for tag, model in models.items():
            with torch.no_grad():
                _ = model.sample_actions(device, obs_on_device, noise=shared_noise)
            activations = model.get_current_step_activations()

            buffers[tag].append(
                {
                    "batch_idx": batch_idx,
                    "activations": activations,
                    "expert_actions": expert_actions,
                    "prompt_tokens": prompt_tokens,
                }
            )

            if len(buffers[tag]) >= args.save_group_size:
                start_idx = group_start[tag]
                end_idx = batch_idx
                _save_activations(output_dir, tag, start_idx, end_idx, buffers[tag])
                buffers[tag].clear()
                group_start[tag] = batch_idx + 1

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for tag, buf in buffers.items():
        if buf:
            start_idx = group_start[tag]
            end_idx = buf[-1]["batch_idx"]
            _save_activations(output_dir, tag, start_idx, end_idx, buf)



if __name__ == "__main__":
    main(tyro.cli(Args))

