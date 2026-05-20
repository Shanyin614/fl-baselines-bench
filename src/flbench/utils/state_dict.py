"""Utilities for PyTorch state_dict arithmetic."""
from __future__ import annotations

import copy
from typing import Iterable, Mapping

import torch

StateDict = dict[str, torch.Tensor]


def clone_state(state: Mapping[str, torch.Tensor]) -> StateDict:
    return {k: v.detach().cpu().clone() for k, v in state.items()}


def state_delta(new_state: Mapping[str, torch.Tensor], base_state: Mapping[str, torch.Tensor]) -> StateDict:
    return {k: new_state[k].detach().cpu().float() - base_state[k].detach().cpu().float() for k in base_state}


def apply_delta(base_state: Mapping[str, torch.Tensor], delta: Mapping[str, torch.Tensor]) -> StateDict:
    return {k: base_state[k].detach().cpu().float() + delta[k].detach().cpu().float() for k in base_state}


def weighted_average_states(states: list[Mapping[str, torch.Tensor]], weights: list[int | float]) -> StateDict:
    if len(states) == 0:
        raise ValueError("weighted_average_states requires at least one state")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    avg: StateDict = {}
    for key in states[0]:
        acc = torch.zeros_like(states[0][key], dtype=torch.float32, device="cpu")
        for state, weight in zip(states, weights):
            acc += state[key].detach().cpu().float() * (float(weight) / total)
        avg[key] = acc
    return avg


def weighted_average_deltas(deltas: list[Mapping[str, torch.Tensor]], weights: list[int | float]) -> StateDict:
    if len(deltas) == 0:
        raise ValueError("weighted_average_deltas requires at least one delta")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    avg: StateDict = {}
    for key in deltas[0]:
        acc = torch.zeros_like(deltas[0][key], dtype=torch.float32, device="cpu")
        for delta, weight in zip(deltas, weights):
            acc += delta[key].detach().cpu().float() * (float(weight) / total)
        avg[key] = acc
    return avg


def perturb_state(state: Mapping[str, torch.Tensor], sigma: float, seed: int) -> StateDict:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    out: StateDict = {}
    for key, value in state.items():
        value_cpu = value.detach().cpu().float().clone()
        if sigma > 0 and value_cpu.is_floating_point():
            noise = torch.randn(value_cpu.shape, generator=generator, dtype=value_cpu.dtype) * sigma
            out[key] = value_cpu + noise
        else:
            out[key] = value_cpu
    return out


def flatten_delta(delta: Mapping[str, torch.Tensor]) -> torch.Tensor:
    parts = [v.detach().cpu().float().reshape(-1) for v in delta.values()]
    if not parts:
        return torch.empty(0)
    return torch.cat(parts)
