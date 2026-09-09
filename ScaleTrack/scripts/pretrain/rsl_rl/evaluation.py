"""Read-only motion evaluation and clip-weighted summaries, independent of Kit."""

import math

import torch


@torch.inference_mode()
def evaluate_motion_batches(env, policy, command, metric_keys, *, max_steps=1000, progress=None, extra_metrics=None):
    """Evaluate each clip once, excluding the environment's auto-reset step.

    A motion with N frames has N-1 advancing steps before motion_time_out resets
    it. Capped clips get exactly max_steps samples. Metrics follow the existing
    command manager's post-physics, pre-reference-advance alignment.
    """
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    if not metric_keys or len(set(metric_keys)) != len(metric_keys):
        raise ValueError("metric_keys must be nonempty and unique")
    if command.num_motion < 1 or env.num_envs < 1:
        raise ValueError("Evaluation requires motions and environments")
    frame_counts = command.time_totals.cpu()
    if torch.any(frame_counts < 2):
        raise ValueError("Each motion requires at least two frames")
    order = torch.argsort(frame_counts, stable=True)
    records = {}
    for offset in range(0, command.num_motion, env.num_envs):
        ids = order[offset:offset + env.num_envs]
        count = len(ids)
        command.motion_ids[:count] = ids
        command.motion_ids[count:] = ids[0]
        command.time_steps.zero_()
        command._future_manual_cache.clear()
        lengths = torch.clamp(frame_counts[ids] - 1, max=max_steps)
        lengths_device = lengths.to(env.device)
        sums = torch.zeros((count, len(metric_keys)), dtype=torch.float64, device=env.device)
        maxima = torch.full_like(sums, -torch.inf)
        obs, _ = env.reset()
        for step in range(int(lengths.max())):
            active = step < lengths_device
            actions = policy(obs.to(env.device))
            obs, _, dones, _ = env.step(actions.to(env.device))
            if torch.any(dones[:count].to(env.device).bool() & active):
                raise RuntimeError("Unexpected reset during a valid evaluation step; refusing biased metrics")
            metrics = command.metrics if extra_metrics is None else {**command.metrics, **extra_metrics()}
            values = torch.stack([metrics[key][:count] for key in metric_keys], dim=-1)
            if not torch.isfinite(values[active]).all():
                raise RuntimeError("Evaluation metrics must be finite; refusing an invalid report")
            sums += torch.where(active[:, None], values, 0.0)
            maxima = torch.maximum(maxima, torch.where(active[:, None], values, -torch.inf))
        means = (sums / lengths_device[:, None]).cpu().tolist()
        maxima = maxima.cpu().tolist()
        for local, motion_id in enumerate(ids.tolist()):
            records[motion_id] = {
                "motion_id": motion_id,
                "motion": command.motion_names[motion_id],
                "source_frames": int(frame_counts[motion_id]),
                "evaluated_steps": int(lengths[local]),
                "truncated": int(frame_counts[motion_id]) - 1 > max_steps,
                "metrics": {
                    key: {"mean": means[local][column], "max": maxima[local][column]}
                    for column, key in enumerate(metric_keys)
                },
            }
        if progress is not None:
            progress(len(records), command.num_motion)
    return [records[index] for index in range(command.num_motion)]


def summarize_motion_metrics(rows, *, thresholds=(0.1, 0.2, 0.5)):
    """Thresholds bound a clip's maximum mean-14-body global position error (m)."""
    if not rows or any(not math.isfinite(value) or value <= 0 for value in thresholds):
        raise ValueError("Nonempty rows and finite positive thresholds are required")

    def aggregate(group):
        result = {
            "num_motions": len(group),
            "evaluated_steps": sum(row["evaluated_steps"] for row in group),
            "truncated_motions": sum(row["truncated"] for row in group),
            "mean_metrics": {
                key: sum(row["metrics"][key]["mean"] for row in group) / len(group)
                for key in group[0]["metrics"]
            },
            "thresholds": {},
        }
        for threshold in thresholds:
            failed = [
                row["motion"] for row in group
                if row["metrics"]["error_body_pos_g"]["max"] > threshold
            ]
            result["thresholds"][str(float(threshold))] = {
                "failed_motions": failed,
                "success_rate": 1.0 - len(failed) / len(group),
            }
        return result

    summary = aggregate(rows)
    datasets = sorted({row["motion"].split("/", 1)[0] for row in rows})
    summary["by_dataset"] = {
        dataset: aggregate([row for row in rows if row["motion"].split("/", 1)[0] == dataset])
        for dataset in datasets
    }
    return summary
