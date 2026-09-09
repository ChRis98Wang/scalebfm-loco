"""Bounded real-physics checks: gravity, resting contact, sliding and subset reset."""

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
sim = None
try:
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from scaletrack.tasks.tracking.tracking_env_cfg import MySceneCfg

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/pretrain/rsl_rl"))
    from target_object import TargetObjectSpec, attach_target_object, reset_target_object

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args.device))
    cfg = SimpleNamespace(scene=InteractiveSceneCfg(num_envs=2, env_spacing=4.0), events=SimpleNamespace())
    cfg.scene.terrain = MySceneCfg(num_envs=2, env_spacing=4.0).terrain
    spec = TargetObjectSpec(position=(0.9, -0.7, 0.8))
    attach_target_object(cfg, spec)
    scene = InteractiveScene(cfg.scene)
    sim.reset()
    env = SimpleNamespace(scene=scene, num_envs=2, device=args.device)
    obj = scene["target_object"]

    def tensor(value):
        return getattr(value, "torch", value)

    def advance(steps):
        for _ in range(steps):
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim.get_physics_dt())

    reset_target_object(env)
    advance(400)
    local = tensor(obj.data.root_pos_w) - scene.env_origins
    assert torch.isfinite(local).all()
    resting_z = local[:, 2].clone()
    assert torch.all((resting_z > 0.14) & (resting_z < 0.16)), resting_z
    assert torch.all(torch.linalg.vector_norm(tensor(obj.data.root_lin_vel_w), dim=-1) < 0.05)
    start_x = local[:, 0].clone()
    velocity = torch.zeros((2, 6), device=args.device)
    velocity[:, 0] = 1.0
    obj.write_root_velocity_to_sim(velocity)
    advance(400)
    displacement = (tensor(obj.data.root_pos_w) - scene.env_origins)[:, 0] - start_x
    assert torch.all((displacement > 0.01) & (displacement < 0.5)), displacement
    final_speed = torch.linalg.vector_norm(tensor(obj.data.root_lin_vel_w), dim=-1)
    assert torch.all(final_speed < 0.05), final_speed
    before_reset = tensor(obj.data.root_pos_w).clone()
    reset_target_object(env, torch.tensor([1], device=args.device))
    reset_position = tensor(obj.data.root_pos_w) - scene.env_origins
    torch.testing.assert_close(reset_position[1], torch.tensor(spec.position, device=args.device), atol=1e-5, rtol=0)
    torch.testing.assert_close(tensor(obj.data.root_pos_w)[0], before_reset[0])
    assert torch.all(tensor(obj.data.root_lin_vel_w)[1] == 0)
    print("[BFM TARGET PHYSICS] PASS " + json.dumps({
        "resting_height_m": resting_z.cpu().tolist(),
        "slide_distance_m_from_1mps": displacement.cpu().tolist(),
        "final_speed_mps": final_speed.cpu().tolist(),
        "subset_reset": True, "calibrated": False,
    }), flush=True)
except BaseException:
    import traceback
    traceback.print_exc()
    print("[BFM TARGET PHYSICS] FAIL", flush=True)
    raise
finally:
    if sim is not None:
        sim.stop()
    app.close(exit_code=1 if sys.exc_info()[0] else 0)
