from collections import OrderedDict
import pdb
import gym.spaces
import numpy as np
import copy
from cs285.networks.policies import MLPPolicy
import gym
import cv2
from cs285.infrastructure import pytorch_util as ptu
from typing import Dict, Tuple, List
import pandas as pd
import matplotlib.pyplot as plt


############################################
############################################


def sample_trajectory(
    env: gym.Env, policy: MLPPolicy, max_length: int, render: bool = False
) -> Dict[str, np.ndarray]:
    """Sample a rollout in the environment from a policy."""
    ob = env.reset()
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    while True:
        # render an image
        if render:
            if hasattr(env, "sim"):
                img = env.sim.render(camera_name="track", height=500, width=500)[::-1]
            else:
                img = env.render(mode="single_rgb_array")
            image_obs.append(
                cv2.resize(img, dsize=(250, 250), interpolation=cv2.INTER_CUBIC)
            )

        # TODO use the most recent ob and the policy to decide what to do
        ac: np.ndarray = policy.get_action(ob)

        # TODO: use that action to take a step in the environment
        next_ob, rew, done, _ = None, None, None, None
        next_ob, rew, done, _ = env.step(ac)

        # TODO rollout can end due to done, or due to max_length
        steps += 1
        rollout_done: bool = None

        # record result of taking that action
        obs.append(ob)
        acs.append(ac)
        rewards.append(rew)
        next_obs.append(next_ob)
        rollout_done = ((len(rewards) == max_length) or done)
        terminals.append(rollout_done)

        ob = next_ob  # jump to next timestep

        # end the rollout if the rollout ended
        if rollout_done:
            break

    return {
        "observation": np.array(obs, dtype=np.float32),
        "image_obs": np.array(image_obs, dtype=np.uint8),
        "reward": np.array(rewards, dtype=np.float32),
        "action": np.array(acs, dtype=np.float32),
        "next_observation": np.array(next_obs, dtype=np.float32),
        "terminal": np.array(terminals, dtype=np.float32),
    }


def sample_trajectories(
    env: gym.Env,
    policy: MLPPolicy,
    min_timesteps_per_batch: int,
    max_length: int,
    render: bool = False,
) -> Tuple[List[Dict[str, np.ndarray]], int]:
    """Collect rollouts using policy until we have collected min_timesteps_per_batch steps."""
    timesteps_this_batch = 0
    trajs = []
    while timesteps_this_batch < min_timesteps_per_batch:
        # collect rollout
        traj = sample_trajectory(env, policy, max_length, render)
        trajs.append(traj)

        # count steps
        timesteps_this_batch += get_traj_length(traj)
    return trajs, timesteps_this_batch


def sample_n_trajectories(
    env: gym.Env, policy: MLPPolicy, ntraj: int, max_length: int, render: bool = False
):
    """Collect ntraj rollouts."""
    trajs = []
    for _ in range(ntraj):
        # collect rollout
        traj = sample_trajectory(env, policy, max_length, render)
        trajs.append(traj)
    return trajs


def compute_metrics(trajs, eval_trajs):
    """Compute metrics for logging."""

    # returns, for logging
    train_returns = [traj["reward"].sum() for traj in trajs]
    eval_returns = [eval_traj["reward"].sum() for eval_traj in eval_trajs]

    # episode lengths, for logging
    train_ep_lens = [len(traj["reward"]) for traj in trajs]
    eval_ep_lens = [len(eval_traj["reward"]) for eval_traj in eval_trajs]

    # decide what to log
    logs = OrderedDict()
    logs["Eval_AverageReturn"] = np.mean(eval_returns)
    logs["Eval_StdReturn"] = np.std(eval_returns)
    logs["Eval_MaxReturn"] = np.max(eval_returns)
    logs["Eval_MinReturn"] = np.min(eval_returns)
    logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

    logs["Train_AverageReturn"] = np.mean(train_returns)
    logs["Train_StdReturn"] = np.std(train_returns)
    logs["Train_MaxReturn"] = np.max(train_returns)
    logs["Train_MinReturn"] = np.min(train_returns)
    logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

    return logs


def convert_listofrollouts(trajs):
    """
    Take a list of rollout dictionaries and return separate arrays, where each array is a concatenation of that array
    from across the rollouts.
    """
    observations = np.concatenate([traj["observation"] for traj in trajs])
    actions = np.concatenate([traj["action"] for traj in trajs])
    next_observations = np.concatenate([traj["next_observation"] for traj in trajs])
    terminals = np.concatenate([traj["terminal"] for traj in trajs])
    concatenated_rewards = np.concatenate([traj["reward"] for traj in trajs])
    unconcatenated_rewards = [traj["reward"] for traj in trajs]
    return (
        observations,
        actions,
        next_observations,
        terminals,
        concatenated_rewards,
        unconcatenated_rewards,
    )


def get_traj_length(traj):
    return len(traj["reward"])


def plot_multiple_experiments(csv_file_paths, labels, x_label, y_label, title, num_xticks=20):
    """
    Plots multiple TensorBoard CSV experiment results on the same graph for comparison.

    Args:
        csv_file_paths (list): List of CSV file paths.
        labels (list): List of labels corresponding to each experiment.
        x_label (str): Label for the X-axis.
        y_label (str): Label for the Y-axis.
        title (str): Graph title.
        num_xticks (int): Number of x-axis ticks to display (default: 20).
    """

    plt.figure(figsize=(16, 5))  # Set figure size

    all_steps = []  # Store all x-values for setting xticks

    for i, csv_file_path in enumerate(csv_file_paths):
        # Load CSV file
        df = pd.read_csv(csv_file_path)

        # Store x-values for tick control
        all_steps.append(df["Step"].values)

        # Plot each experiment with a different label
        plt.plot(df["Step"], df["Value"], marker='o', linestyle='-', label=labels[i])

    # Set more X-axis ticks (forcing more frequent labels)
    min_x = min(min(steps) for steps in all_steps)
    max_x = max(max(steps) for steps in all_steps)
    xtick_values = np.linspace(min_x, max_x, num=num_xticks, dtype=int)  # Generate evenly spaced ticks
    plt.xticks(xtick_values)  # Set ticks on x-axis

    # Customize the graph
    plt.xlabel(x_label)  
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()  # Show labels for each experiment

    # Show the plot
    plt.show()
