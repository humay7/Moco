import os
import jax
import jax.numpy as jnp
import numpy as np
from jax import disable_jit
from functools import partial
import rlax
import tqdm
import matplotlib.pyplot as plt
from functools import partial
import argparse
import chex
from chex import dataclass, PRNGKey, Array
from flax.training.early_stopping import EarlyStopping
import orbax.checkpoint as ocp
import haiku as hk
import optax
import uuid
# import jmp

from datetime import datetime

from learned_optimization.outer_trainers import full_es, truncated_pes, truncated_es, gradient_learner, truncation_schedule, lopt_truncated_step

from learned_optimization.tasks import quadratics, base as tasks_base
from learned_optimization.tasks.fixed import image_mlp
from learned_optimization.tasks.datasets import base as datasets_base

from learned_optimization.learned_optimizers import base as lopt_base, mlp_lopt
from learned_optimization.optimizers import learning_rate_schedules, base as opt_base
from learned_optimization.optimizers.optax_opts import Adam, SGD, SGDM, RMSProp, AdamW, OptaxOptimizer
from learned_optimization.outer_trainers.gradient_learner import MetaInitializer

from learned_optimization import optimizers, eval_training
# import oryx
# from learned_optimization import summary
# assert summary.ORYX_LOGGING, "Oryx logging not working"

from moco.environments import CustomTSP as TSP
from moco.data_utils import *
from moco.plot_utils import plot_tsp_grid
from moco.tsp_actors import nearest_neighbor, HeatmapActor
from moco.rl_utils import random_actor, greedy_actor, rollout, random_initial_position, greedy_rollout, pomo_rollout
from moco.utils import *
from moco.tasks import TspTaskFamily, train_task, train_task_with_trajectory, TspTaskParams
from moco.lopt import HeatmapOptimizer
from moco.ddpg_agent import DDPGAgent
import mlflow


if __name__ == "__main__":

    ################## config ##################
    parser = argparse.ArgumentParser()
    # TaskFamily and problem setting
    parser.add_argument("--problem_size", type=int)
    parser.add_argument("--task_batch_size", type=int, help="b in the paper")
    parser.add_argument("--max_length", type=int, default=50, help="Budget K in the paper") # number of steps to unroll the optimizer on the inner task
    parser.add_argument("--episodes_per_batch", type=int, default=4, help="Number of episodes to collect per batch")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--top_k", type=int, default=32)
    parser.add_argument("--heatmap_init_strategy", type=str, choices=["heuristic", "constant"], default="heuristic") # doesnt get used, the initialization is overwritten in the learned optimizer
    parser.add_argument("--rollout_actor", type=str, choices=["softmax", "entmax"], default="softmax")
    parser.add_argument("--two_opt_t_max", type=int, default=None)
    parser.add_argument("--first_accept", action="store_true")

    # pgl calculation
    parser.add_argument("--causal", "-c", help="use causal accumulation of rewards for policy gradient calc", action="store_true")
    parser.add_argument("--baseline", "-b", help="specify baseline for policy gradient calc", type=str, default="avg", choices=[None, "avg"])

    # meta loss
    parser.add_argument("--meta_loss_type", type=str, choices=["best", "log"], default="best")

    # DDPG specific parameters
    parser.add_argument("--actor_lr", type=float, default=1e-4)
    parser.add_argument("--critic_lr", type=float, default=1e-3)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--gamma", type=float, default=0.99)

    # metaTraining
    parser.add_argument("--parallel_tasks_train", type=int)
    parser.add_argument("--outer_lr", type=float)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--outer_train_steps", type=int, default=10000)
    parser.add_argument("--trunc_schedule", type=str, choices=["constant", "loguniform", "piecewise_linear"], default="constant") # for full es and pes training
    parser.add_argument("--min_length", type=int, default=10)
    parser.add_argument("--piecewise_linear_fraction", type=float, default=0.2, help="fraction of outer_train_steps after which the truncation length is max_length")
    parser.add_argument("--patience", type=int, default=20) # early stopping
    parser.add_argument("--dont_stack_antithetic", action="store_true") # whether to stack antithetic samples for gradient estimation
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--lr_schedule", type=str, choices=["constant", "cosine"], default="cosine")
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--num_devices", type=int, default=None)
    parser.add_argument("--clip_loss_diff", type=float, default=None)
    parser.add_argument("--sigma", type=float, default=0.01)

    # meta optimizer
    parser.add_argument("--update_strategy", type=str, choices=["direct", "temperature"], default="temperature")
    parser.add_argument("--aggregation", type=str, choices=["sum", "max"], default="sum")
    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--num_layers_init", type=int, default=3)
    parser.add_argument("--num_layers_update", type=int, default=3)
    parser.add_argument("--normalization", type=str, choices=["pre", "post", "none"], default="post")
    parser.add_argument("--checkpoint_folder", "-cf", help="folder to load checkpoint from", type=str, default=None)

    # validation and logging
    parser.add_argument("--parallel_tasks_val", type=int)
    parser.add_argument("--val_path", type=str)
    parser.add_argument("--model_save_path", type=str)
    parser.add_argument("--val_steps", type=int, default=200)
    parser.add_argument("--log_steps", type=int, default=1)
    parser.add_argument("--mlflow_uri", type=str, default="logs")
    parser.add_argument("--experiment_name", type=str, default="ddpg_tsp")
    parser.add_argument("--disable_tqdm", default=False, action="store_true")
    parser.add_argument("--ood_path", default=None, type=str)

    # debug
    parser.add_argument("--disable_jit", default=False, action="store_true")
    parser.add_argument("--subset", type=parse_slice, default=None, help="slice of the val set to use for validation (e.g. 0:16) only for debugging")

    args = parser.parse_args()

    # current time as string for saving in human readable format
    # start_time = datetime.now().strftime("%m%d%Y-%H%M%S")
    if args.model_save_path is not None:
        unique_filename = str(uuid.uuid4())
        args.model_save_path = os.path.join(args.model_save_path, unique_filename)

    if args.num_devices is None:
        args.num_devices = len(jax.devices())

    assert args.min_length <= args.max_length, "loguniform_trunc_min must be smaller equal than max_length"
    assert args.parallel_tasks_train % args.num_devices == 0, f"parallel_tasks_train must be divisible by num_devices {args.num_devices}, jax_devices: {jax.devices()}"
    assert args.parallel_tasks_val % args.num_devices == 0, f"parallel_tasks_val must be divisible by num_devices {args.num_devices}, jax_devices: {jax.devices()}"
    
    # test gpu # TODO: switch to chex 
    print("jax has gpu:", jax_has_gpu())
    ################## config ##################

    val_dataset = load_data(args.val_path, batch_size=args.parallel_tasks_val, subset=args.subset)
    task_family = TspTaskFamily(args.problem_size, args.task_batch_size, args.k, baseline = args.baseline, causal = args.causal, meta_loss_type = args.meta_loss_type, top_k=args.top_k, heatmap_init_strategy=args.heatmap_init_strategy, rollout_actor=args.rollout_actor, two_opt_t_max=args.two_opt_t_max, first_accept=args.first_accept)

    # Initialize DDPG agent instead of ES
    # We'll create a dummy observation to infer the correct feature sizes
    # First, let's create a dummy task to get the correct graph structure
    dummy_observation = task_family.dummy_model_state()
    
    # load from checkpoint if available
    if args.checkpoint_folder is not None:
        # Load DDPG state from checkpoint
        restore_options = ocp.CheckpointManagerOptions(
            best_mode='min',
            best_fn=lambda x: x['val_last_best_reward'] if 'val_last_best_reward' in x else x['val_gap_avg'],
        )
        restore_mngr = ocp.CheckpointManager(
            args.checkpoint_folder,
            ocp.PyTreeCheckpointer(),
            options=restore_options)
        
        metadata = restore_mngr.metadata()
        # Overwrite args with metadata from checkpoint that affects the optimizer
        args.embedding_size = metadata['embedding_size']
        args.num_layers_init = metadata['num_layers_init']
        args.num_layers_update = metadata['num_layers_update']
        args.aggregation = metadata['aggregation']
        args.normalization = metadata['normalization']
        print("Warning: Overwriting args with metadata from checkpoint that affects the optimizer")
        
        # Create DDPG agent without dummy observation (use hardcoded feature sizes for checkpoint compatibility)
        ddpg_agent = DDPGAgent(
            update_strategy=args.update_strategy,
            embedding_size=args.embedding_size,
            num_layers_init=args.num_layers_init,
            num_layers_update=args.num_layers_update,
            aggregation=args.aggregation,
            normalization=args.normalization,
            dummy_observation=None,  # Don't use dummy observation for checkpoint loading
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            tau=args.tau,
            gamma=args.gamma
        )
        
        # Load the DDPG state
        ddpg_agent.state = restore_mngr.restore(restore_mngr.best_step())
        print(f"Loaded DDPG agent from checkpoint {args.checkpoint_folder} step {restore_mngr.best_step()}")
    else:
        # Initialize from scratch with dummy observation
        ddpg_agent = DDPGAgent(
            update_strategy=args.update_strategy,
            embedding_size=args.embedding_size,
            num_layers_init=args.num_layers_init,
            num_layers_update=args.num_layers_update,
            aggregation=args.aggregation,
            normalization=args.normalization,
            dummy_observation=dummy_observation,  # Use dummy observation to infer feature sizes
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            tau=args.tau,
            gamma=args.gamma
        )
        
        # Initialize from scratch
        key = jax.random.PRNGKey(0)
        ddpg_agent.state = ddpg_agent.init(key)
        print("Initialized DDPG agent from scratch")

    # Create global HeatmapOptimizer (reused across functions) - same pattern as tsp_meta_train.py
    heatmap_optimizer = HeatmapOptimizer(
        embedding_size=args.embedding_size,
        num_layers_init=args.num_layers_init,
        num_layers_update=args.num_layers_update,
        aggregation=args.aggregation,
        update_strategy=args.update_strategy,
        normalization=args.normalization,
        dummy_observation=dummy_observation  # Use same dummy observation for consistency
    )

    # Keeps a maximum of 3 checkpoints and keeps the best one
    options = ocp.CheckpointManagerOptions(
        max_to_keep=3,
        best_mode='min',
        best_fn=lambda x: x['val_last_best_reward'],
    )
    metadata = dict(vars(args))
    metadata['subset'] = str(metadata['subset']) # to avoid json serialization error
    metadata['lopt'] = 'gnn'  # Required for evaluation
    mngr = ocp.CheckpointManager(
        args.model_save_path,
        ocp.PyTreeCheckpointer(),
        options=options,
        metadata=metadata)

    early_stop = EarlyStopping(min_delta=1e-3, patience=args.patience)
    print("early stop:", early_stop)
    print("mngr:", mngr)
    
    # debug train_task
    # problem = next(val_dataset.as_numpy_iterator())[0]
    # res = train_task(theta, jnp.array(problem), key)
    # print(res)
    train_task_batched = jax.vmap(train_task, in_axes=(0, 0, None, None, None))
    train_task_jit = jax.jit(train_task_batched, static_argnames=['num_steps', 'optimizer', 'task_family'])
    if args.num_devices > 1:
        train_task_pmap = jax.pmap(train_task_batched, axis_name='data', in_axes=(0, 0, None, None, None), static_broadcasted_argnums=(2,3,4))
    

    @partial(jax.jit, static_argnames=['task_family', 'heatmap_optimizer', 'max_length'])
    def collect_episodes_batched(task_family,
                                 actor_params,
                                 problems,         # (E, problem_size, 2) or similar
                                 keys,             # (E,)
                                 max_length=50,
                                 heatmap_optimizer=None):
        """
        Collect E episodes of length T=max_length in parallel and return a flat batch:
          states_flat, actions_flat, rewards_flat, next_states_flat, dones_flat
        where states/next_states are GraphsTuple pytrees with leaves shaped (E*T, ...).
        """

        # ---- helpers -----------------------------------------------------------

        def _concat_if_dict(x):
            """Concatenate dict-of-arrays along -1 in a stable (sorted keys) order."""
            if isinstance(x, dict):
                parts = [x[k] for k in sorted(x.keys())]
                chex.assert_equal_shape_prefix(parts, prefix_len=1)
                return jnp.concatenate(parts, axis=-1)
            return x

        def _ensure_array_features(graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
            """Return a GraphsTuple whose nodes/edges/globals are arrays, not dicts."""
            return graph._replace(
                nodes=_concat_if_dict(graph.nodes),
                edges=_concat_if_dict(graph.edges),
                globals=_concat_if_dict(graph.globals),
            )

        def _time_shift_next(tree):
            """Create s_{t+1} by shifting along time dim: (E,T,...) -> (E,T,...)."""
            return jax.tree_map(lambda x: jnp.concatenate([x[:, 1:], x[:, -1:]], axis=1), tree)

        def _merge_ET(tree, E, T):
            """Flatten (E,T,...) -> (E*T,...) for every leaf."""
            return jax.tree_map(lambda x: x.reshape((E * T, *x.shape[2:])), tree)

        # ---- single-episode collection ----------------------------------------

        def collect_single_episode(problem, key):
            # Build optimizer with current actor params
            opt_fn = heatmap_optimizer.opt_fn(actor_params)

            # Task params from coordinates (start node arbitrary constant here)
            task_params = TspTaskParams(coordinates=problem, starting_node=1)

            # Run trajectory-producing inference
            traj = train_task_with_trajectory(task_params, key, max_length, opt_fn, task_family)

            # Rewards from improvements in best solution (minimization -> negative deltas)
            best_rewards = traj['best_reward']                         # (T,)
            rewards = jnp.zeros(max_length, dtype=jnp.float32)
            rewards = rewards.at[0].set(-best_rewards[0])
            if max_length > 1:
                improvements = best_rewards[:-1] - best_rewards[1:]    # prev - curr
                rewards = rewards.at[1:].set(improvements.astype(jnp.float32))

            # States/actions along time
            states  = traj['state']    # (T, ...) GraphsTuple with dict or arrays in leaves
            actions = traj['action']   # (T, ...) array (e.g., edge heatmaps)

            # Convert dict features -> arrays per time step
            # states is (T, ...); vmap over time then add episode axis later
            states = jax.vmap(_ensure_array_features)(states)  # (T, ...)

            # Done flags (only last step is terminal)
            dones = jnp.zeros(max_length, dtype=jnp.float32).at[-1].set(1.0)

            return states, actions, rewards, states, dones  # next_states will be time-shifted later

        # ---- batch episodes with vmap(E) --------------------------------------

        batched_collect = jax.vmap(collect_single_episode, in_axes=(0, 0))
        states, actions, rewards, next_states, dones = batched_collect(problems, keys)  # (E, T, ...)

        # Build proper next_states by shifting time within each episode
        # First, ensure array features for the batched GraphsTuple (E,T,...)
        states = jax.tree_map(lambda x: x, states)          # no-op; keeps structure explicit
        states = states._replace(                           # ensure array features across (E,T,...)
            nodes=_concat_if_dict(states.nodes),
            edges=_concat_if_dict(states.edges),
            globals=_concat_if_dict(states.globals),
        )
        next_states = _time_shift_next(states)

        # Sanity: actions must align with edges on leading dims (E,T,...)
        chex.assert_equal_shape_prefix([states.edges, actions], prefix_len=2)
        chex.assert_equal_shape_prefix([next_states.edges, actions], prefix_len=2)

        # Flatten (E,T) -> (E*T) per leaf/array
        E, T = actions.shape[:2]
        states_flat      = _merge_ET(states, E, T)
        next_states_flat = _merge_ET(next_states, E, T)

        # Flatten arrays
        if actions.ndim >= 3:
            actions_flat = actions.reshape(E * T, *actions.shape[2:])
        else:
            actions_flat = actions.reshape(E * T)

        rewards_flat = jnp.asarray(rewards, dtype=jnp.float32).reshape(E * T)
        dones_flat   = jnp.asarray(dones,   dtype=jnp.float32).reshape(E * T)
        
        return (states_flat, actions_flat, rewards_flat, next_states_flat, dones_flat)

    def validate(dataset, task_family, actor_params, key, aggregate=True):
        """Validate the DDPG agent on a batch of problems from the validation set."""
        # Use global HeatmapOptimizer with current actor parameters
        opt = heatmap_optimizer.opt_fn(actor_params)
        
        metrics = []
        for i, batch in enumerate(dataset.as_numpy_iterator()):
            key, subkey = jax.random.split(key)
            keys = jax.random.split(subkey, batch.shape[0])
            
            # print(jax.tree_map(lambda x: x.shape, (actor_params, batch, keys)))
            task_p_fn = jax.vmap(lambda c,s: TspTaskParams(coordinates=c, starting_node=s), in_axes=(0,0))
            batched_task_params = task_p_fn(jnp.array(batch, dtype=jnp.float32), jnp.ones((batch.shape[0],), dtype=jnp.int32))

            if args.num_devices > 1 and batch.shape[0] % args.num_devices == 0: # only use pmap if batch size is divisible by num_devices, usually the last batch is not and then we use jit on single device
                res = unsplit(train_task_pmap(split(batched_task_params), split(keys), args.max_length, opt, task_family))
                print(f"pmap with {args.num_devices} devices and batch size {batch.shape[0]}")
            else:
                res = train_task_jit(batched_task_params, keys, args.max_length, opt, task_family)
                print("jit with 1 device and batch size", batch.shape[0])
            metrics.append(res)
        # aggregate batches of results
        results = {key:jnp.mean(jnp.concatenate([val[key] for val in metrics], axis=0), axis=0) for key in metrics[0].keys()}
        if aggregate:
            results = {
                **{f'val_mean_{key}':jnp.mean(results[key]).item() for key in results}, 
                **{f'val_last_{key}':results[key][-1].item() for key in results}
            }
        return results
    
    def split(tree):
        """Splits the first axis of `arr` evenly across the number of devices."""
        return jax.tree_map(lambda arr: arr.reshape(args.num_devices, arr.shape[0] // args.num_devices, *arr.shape[1:]), tree)

    def unsplit(tree):
        """Concatenates the first axis of `arr` across all devices."""
        return jax.tree_map(lambda arr: arr.reshape(-1, *arr.shape[2:]), tree)
    
    # Initialize random key
    key = jax.random.PRNGKey(0)
    
    # print args
    print("args:")
    for arg in vars(args):
        print(arg, getattr(args, arg))

    print("Starting DDPG training...", flush=True)
    mlflow.set_tracking_uri(uri=args.mlflow_uri)
    experiment = mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(experiment_id=experiment.experiment_id) as mlflow_cm, disable_jit(args.disable_jit) as jit_cm:

        # log args
        mlflow.log_params(vars(args))
        slurm_job_id = os.environ.get('SLURM_JOB_ID', None)
        if slurm_job_id is not None:
            mlflow.log_param('slurm_job_id', slurm_job_id)
            print(f'Logged slurm_job_id: {slurm_job_id}', flush=True)

        for i in tqdm.tqdm(range(args.outer_train_steps), disable=args.disable_tqdm):
            # validation
            if i % args.val_steps == 0:
                key, subkey = jax.random.split(key)
                actor_params = ddpg_agent.get_actor_params(ddpg_agent.state)
                results = validate(val_dataset, task_family, actor_params, subkey)
                mlflow.log_metrics(results, step=i)

                # checkpointing
                mngr.save(i, ddpg_agent.state, metrics=results)

                # early stopping
                early_stop_score = results['val_last_best_reward']
                _, early_stop = early_stop.update(early_stop_score)
                if early_stop.should_stop:
                    print('Met early stopping criteria, breaking...')
                    break

            # DDPG training - collect episodes and update
            actor_params = ddpg_agent.get_actor_params(ddpg_agent.state)
            
            # Collect episodes in parallel using batched collection
            key, subkey = jax.random.split(key)
            keys = jax.random.split(subkey, args.episodes_per_batch)
            
            # Sample problems in parallel
            problems = jnp.array([task_family.sample(k).coordinates for k in keys])
            
            # Collect episodes in parallel and get training batch directly
            batch = collect_episodes_batched(
                task_family, actor_params, problems, keys, args.max_length, heatmap_optimizer
            )
            
            # Update DDPG agent
            ddpg_agent.state, metrics = ddpg_agent.update(ddpg_agent.state, batch)
            
            if i % args.log_steps == 0:
                # replace '||' with '_' to make it a valid mlflow metric name
                metrics = {k.replace("||", "__"): float(v) for k, v in metrics.items() if not 'collect' in k}
                mlflow.log_metrics(metrics, step=i)

        # training done, log final validation metrics
        best_val_parameters = ddpg_agent.get_actor_params(mngr.restore(mngr.best_step()))
        key, subkey = jax.random.split(key)
        
        results = validate(val_dataset, task_family, best_val_parameters, subkey, aggregate=False)
        aggregates = {
                **{f'val_mean_{key}':jnp.mean(results[key]).item() for key in results}, 
                **{f'val_last_{key}':results[key][-1].item() for key in results}
            }
        mlflow.log_metrics(aggregates, step=args.outer_train_steps)

        # log ood
        if args.ood_path is not None:
            ood_dataset = load_data(args.ood_path, batch_size=args.parallel_tasks_val, subset=args.subset)
            _, ood_size, _ = ood_dataset.element_spec.shape
            ood_family = TspTaskFamily(ood_size, args.task_batch_size, args.k, baseline = args.baseline, causal = args.causal, meta_loss_type = args.meta_loss_type, top_k=args.top_k, two_opt_t_max=args.two_opt_t_max, first_accept=args.first_accept)
            key, subkey = jax.random.split(key)
            ood_results = validate(ood_dataset, ood_family, best_val_parameters, subkey, aggregate=True)
            mlflow.log_metrics({'ood_score': ood_results['val_last_best_reward']}, step=0)
