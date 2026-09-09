import os
import sys
import time
import signal
from pathlib import Path
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig
from loguru import logger

# Source-checkout execution does not require installing hardware/TensorRT extras.
for source in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))


def validate_runtime_config(cfg):
    target = str(cfg.simulator.get("_target_", ""))
    allowed = {"simulator.mujoco_simulator.MujocoSimulator", "scalebridge.simulator.mujoco_simulator.MujocoSimulator"}
    if target not in allowed and cfg.get("allow_real_world", False) is not True:
        raise ValueError("Only MuJoCo is enabled by default; real-world backends need separate explicit opt-in")
    steps = cfg.get("max_steps", 0)
    if type(steps) is not int or steps < 0:
        raise ValueError("max_steps must be a nonnegative integer")
    if cfg.simulator.config.get("headless", False) and steps < 1:
        raise ValueError("Headless simulation requires an explicit positive max_steps bound")
    warmup = cfg.get("warmup_steps", 20)
    if type(warmup) is not int or not 0 <= warmup <= 100:
        raise ValueError("warmup_steps must be an integer in 0..100")
    return steps, warmup


def run_controller(cfg):
    max_steps, warmup = validate_runtime_config(cfg)
    agent = env = None
    previous = signal.getsignal(signal.SIGTERM)

    def terminate(signum, frame):
        raise KeyboardInterrupt("ScaleBridge service stopped")

    signal.signal(signal.SIGTERM, terminate)
    try:
        agent = instantiate(cfg.agent)
        env = instantiate(cfg.env, metadata_dict=agent.get_meta_data())
        obs_dict = env.reset()
        for _ in range(warmup):
            agent.get_action(obs_dict)
        steps = 0
        while not max_steps or steps < max_steps:
            running = getattr(env.simulator, "is_running", lambda: True)
            if not running():
                break
            started = time.monotonic()
            agent.before_step(obs_dict)
            tgt_dof_pos, action = agent.get_action(obs_dict)
            obs_dict = env.step(tgt_dof_pos, action)
            agent.after_step(obs_dict, action)
            steps += 1
            if cfg.get("realtime", True):
                remaining = env.dt - (time.monotonic() - started)
                if remaining > 0:
                    time.sleep(remaining)
        return {"steps": steps, "max_steps": max_steps, "training_updates": 0}
    finally:
        for resource in (env, agent):
            if resource is not None and hasattr(resource, "close"):
                try:
                    resource.close()
                except Exception as error:
                    logger.error(f"[Loop] Resource close failed: {error}")
        signal.signal(signal.SIGTERM, previous)

@hydra.main(
    version_base=None,
    config_path="config",
    config_name="base"
)
def main(cfg: DictConfig) -> None:
    
    hydra_log_path = os.path.join(HydraConfig.get().runtime.output_dir, 'run.log')
    logger.remove()
    logger.add(hydra_log_path, level='DEBUG')

    console_log_level = os.environ.get('LOGURU_LEVEL', 'INFO').upper()
    logger.add(sys.stdout, level=console_log_level, colorize=True)
    logger.info(f'Log saved to {hydra_log_path}')

    result = run_controller(cfg)
    logger.info(f"[Loop] Finished: {result}")


if __name__=="__main__":
    main()
