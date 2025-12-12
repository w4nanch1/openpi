import collections
import dataclasses
import logging
import math
import pathlib
import pickle
from typing import Optional

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 1

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_10"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos
    activations_out_path: Optional[str] = "data/libero/activations"  # Directory path to save activations pkl files. If None, activations are not saved.

    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            episode_activations = []  # Store activations for this episode

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        # Query model to get action
                        inference_result = client.infer(element)
                        action_chunk = inference_result["actions"]
                        
                        # Collect activations if available
                        if "activations" in inference_result and args.activations_out_path:
                            # Convert string keys back to integers for internal processing
                            activations_int_keys = {int(k): v for k, v in inference_result["activations"].items()}
                            episode_activations.append(activations_int_keys)
                        
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1
            # Save activations for this episode immediately
            if args.activations_out_path and episode_activations:
                # Process activations for this episode
                # episode_activations is a list of dicts, each dict is {layer_idx: [num_denoising_steps, hidden_size]}
                # We want to save: {layer_idx: [all_timesteps_combined, hidden_size]}
                episode_activations_dict = {}
                for timestep_activations in episode_activations:
                    for layer_idx, activation_array in timestep_activations.items():
                        if layer_idx not in episode_activations_dict:
                            episode_activations_dict[layer_idx] = []
                        # activation_array shape: [num_denoising_steps, hidden_size]
                        episode_activations_dict[layer_idx].append(activation_array)
                
                # Concatenate all timesteps for each layer
                for layer_idx in episode_activations_dict:
                    # Concatenate all timesteps: [total_timesteps * num_denoising_steps, hidden_size]
                    episode_activations_dict[layer_idx] = np.concatenate(
                        episode_activations_dict[layer_idx], axis=0
                    )
                
                # Save to pkl file: one file per rollout
                base_dir = pathlib.Path(args.activations_out_path)
                # Create directory structure: activations_out_path/task_{task_id}/
                save_dir = base_dir / f"task_{task_id}"
                save_dir.mkdir(parents=True, exist_ok=True)
                
                # File name: episode_{episode_idx}.pkl
                save_path = save_dir / f"episode_{episode_idx}.pkl"
                with open(save_path, "wb") as f:
                    pickle.dump(episode_activations_dict, f)
                
                logging.info(f"Saved activations for task {task_id}, episode {episode_idx} to {save_path}")
                # Log sample structure
                if episode_activations_dict:
                    sample_layer = list(episode_activations_dict.keys())[0]
                    sample_shape = episode_activations_dict[sample_layer].shape
                    logging.info(f"  Activation shape for layer {sample_layer}: {sample_shape}")

            # Save a replay video of the episode
            # Save to the same directory as activations
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            
            if args.activations_out_path:
                # Save to activations directory structure: activations_out_path/task_{task_id}/
                base_dir = pathlib.Path(args.activations_out_path)
                save_dir = base_dir / f"task_{task_id}"
                save_dir.mkdir(parents=True, exist_ok=True)
                
                # File name: rollout_{task_segment}_epoch{episode_idx}_{suffix}.mp4
                video_save_path = save_dir / f"rollout_{task_segment}_epoch{episode_idx}_{suffix}.mp4"
            else:
                # Fallback to video_out_path if activations_out_path is not set
                video_save_path = pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_epoch{episode_idx}_{suffix}.mp4"
            
            imageio.mimwrite(
                video_save_path,
                [np.asarray(x) for x in replay_images],
                fps=10,
            )
            logging.info(f"Saved video to {video_save_path}")

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")
    
    if args.activations_out_path:
        logging.info(f"Activations saved to: {pathlib.Path(args.activations_out_path)}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = tyro.cli(Args)
    eval_libero(args)
