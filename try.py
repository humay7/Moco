import flax.linen as nn
import optax
import jax
import jax.numpy as jnp
import numpy as np
from typing import Sequence


# 1️⃣ CleanRL-style QNetwork (Critic)
class QNetwork(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray, a: jnp.ndarray):
        x = jnp.concatenate([x, a], -1)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x.squeeze(-1)


# 2️⃣ CleanRL-style Actor
class Actor(nn.Module):
    action_dim: int
    action_scale: jnp.ndarray
    action_bias: jnp.ndarray

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        x = nn.tanh(x)
        return x 


# 3️⃣ DDPG Update Step (same as before, with QNetwork as critic)
@jax.jit
def ddpg_update_step(trainer_state, state, reward, next_state, gamma=0.99, tau=0.005):
    actor = trainer_state["actor"]
    critic = trainer_state["critic"]
    actor_params = trainer_state["actor_params"]
    critic_params = trainer_state["critic_params"]
    target_actor_params = trainer_state["target_actor_params"]
    target_critic_params = trainer_state["target_critic_params"]
    actor_opt = trainer_state["actor_opt"]
    critic_opt = trainer_state["critic_opt"]
    actor_opt_state = trainer_state["actor_opt_state"]
    critic_opt_state = trainer_state["critic_opt_state"]

    next_action = actor.apply(target_actor_params, next_state)
    target_q = critic.apply(target_critic_params, next_state, next_action)
    y = reward + gamma * target_q

    def critic_loss_fn(params):
        q = critic.apply(params, state, actor.apply(actor_params, state))
        return (q - y) ** 2

    critic_grads = jax.grad(critic_loss_fn)(critic_params)
    critic_updates, new_critic_opt_state = critic_opt.update(critic_grads, critic_opt_state)
    new_critic_params = optax.apply_updates(critic_params, critic_updates)

    def actor_loss_fn(params):
        action = actor.apply(params, state)
        return -critic.apply(new_critic_params, state, action)

    actor_grads = jax.grad(actor_loss_fn)(actor_params)
    actor_updates, new_actor_opt_state = actor_opt.update(actor_grads, actor_opt_state)
    new_actor_params = optax.apply_updates(actor_params, actor_updates)

    def soft_update(p, tp): return tau * p + (1 - tau) * tp
    new_target_actor_params = jax.tree_map(soft_update, new_actor_params, target_actor_params)
    new_target_critic_params = jax.tree_map(soft_update, new_critic_params, target_critic_params)

    return {
        "actor": actor,
        "critic": critic,
        "actor_params": new_actor_params,
        "critic_params": new_critic_params,
        "target_actor_params": new_target_actor_params,
        "target_critic_params": new_target_critic_params,
        "actor_opt": actor_opt,
        "critic_opt": critic_opt,
        "actor_opt_state": new_actor_opt_state,
        "critic_opt_state": new_critic_opt_state,
    }


# 4️⃣ DDPGGradientEstimator Wrapper
class DDPGGradientEstimator:
    def __init__(self, trainer_state, env, update_fn, num_episodes=100):
        self.state = trainer_state
        self.env = env
        self.update_fn = update_fn
        self.num_episodes = num_episodes

    def run(self):
        for ep in range(self.num_episodes):
            state = self.env.reset()
            for t in range(1):  # Single-step update
                action = self.state["actor"].apply(self.state["actor_params"], state)
                next_state, reward, done, _ = self.env.step(action)
                self.state = self.update_fn(self.state, state, reward, next_state)
                state = next_state
            print(f"[DDPG] Episode {ep + 1} done")


# 5️⃣ ddpg_estimator for MOCO
def ddpg_estimator(task_family):
    truncated_step = lopt_truncated_step.VectorizedLOptTruncatedStep(
        task_family,
        lopt,
        truncation_schedule.NeverEndingTruncationSchedule(),
        num_tasks=args.parallel_tasks_train // args.num_devices,
        meta_loss_with_aux_key="meta_loss",
        task_name=str(task_family)
    )

    # Environment wrapper
    class TruncatedStepEnv:
        def __init__(self, truncated_step):
            self.truncated_step = truncated_step
            self.state = None
            self.best_loss = None

        def reset(self):
            self.state = self.truncated_step.init_meta_params()
            self.best_loss, _ = self.truncated_step.run(self.state)
            return np.array(self.state)

        def step(self, action):
            self.state = jnp.array(action)
            loss, _ = self.truncated_step.run(self.state)
            reward = float(self.best_loss - loss)
            self.best_loss = min(self.best_loss, loss)
            return np.array(self.state), reward, False, {}

    env = TruncatedStepEnv(truncated_step)

    # Actor & Critic (CleanRL style)
    action_dim = task_family.num_edges
    action_scale = jnp.ones((action_dim,))
    action_bias = jnp.zeros((action_dim,))

    actor = Actor(action_dim=action_dim, action_scale=action_scale, action_bias=action_bias)
    critic = QNetwork()

    rng = jax.random.PRNGKey(0)
    dummy_state = np.ones((action_dim,))
    actor_params = actor.init(rng, dummy_state)
    critic_params = critic.init(rng, dummy_state, dummy_state)

    target_actor_params = actor_params
    target_critic_params = critic_params

    actor_opt = optax.adam(1e-3)
    critic_opt = optax.adam(1e-3)

    actor_opt_state = actor_opt.init(actor_params)
    critic_opt_state = critic_opt.init(critic_params)

    trainer_state = {
        "actor": actor,
        "critic": critic,
        "actor_params": actor_params,
        "critic_params": critic_params,
        "target_actor_params": target_actor_params,
        "target_critic_params": target_critic_params,
        "actor_opt": actor_opt,
        "critic_opt": critic_opt,
        "actor_opt_state": actor_opt_state,
        "critic_opt_state": critic_opt_state,
    }

    return DDPGGradientEstimator(trainer_state, env, ddpg_update_step, num_episodes=100)
