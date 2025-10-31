import jax
import jax.numpy as jnp
import jax.random as random
import haiku as hk
import optax
import chex
from chex import Array, ArrayTree, PRNGKey
from typing import Any, Callable, Optional, Tuple, Union
import jraph
from jraph import GraphsTuple
from flax.training.train_state import TrainState
from flax import struct
from flax.core import FrozenDict
from functools import partial
from moco.gnn import GNN

@struct.dataclass
class DDPGTrainState(TrainState):
    """TrainState with target parameters for DDPG"""
    target_params: FrozenDict



@struct.dataclass
class DDPGState:
    """State for DDPG training"""
    actor_state: DDPGTrainState
    critic_state: DDPGTrainState
    step: int

class DDPGAgent:
    """DDPG agent for training HeatmapOptimizer (phi_init and phi_update) via critic feedback.
    
    The actor is the HeatmapOptimizer (external), which generates actions (heatmaps).
    The critic evaluates Q(state, action) where action is concatenated to edge features.
    DDPG updates the HeatmapOptimizer parameters based on critic gradients.
    
    The critic uses only the update network for all timesteps. States at t=0
    have gradient/momentum features that are zero-padded.
    """
    
    def __init__(self, 
                 heatmap_optimizer,  # External HeatmapOptimizer instance (actor)
                 embedding_size=64, 
                 num_layers_init=3, 
                 num_layers_update=3, 
                 aggregation='max', 
                 num_node_features=1, 
                 num_edge_features=41, 
                 num_global_features=45, 
                 normalization="pre", 
                 dummy_observation=None,
                 actor_lr: float = 1e-4,
                 critic_lr: float = 1e-3,
                 tau: float = 0.005,
                 gamma: float = 0.99):
        
        # Store the HeatmapOptimizer (actor)
        self.heatmap_optimizer = heatmap_optimizer
        
        # Network parameters
        self.embedding_size = embedding_size
        self.num_layers_update = num_layers_update
        self.num_layers_init = num_layers_init
        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features
        self.num_global_features = num_global_features
        self.dummy_observation = dummy_observation
        self.aggregation = aggregation
        self.normalization = normalization
        
        # DDPG specific parameters
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.tau = tau
        self.gamma = gamma
        
        # Critic update network - same GNN structure but decode_globals=True, decode_edges=False
        # Action gets concatenated to edge features
        def critic_update_forward(graph):
            network = GNN(
                num_layers=self.num_layers_update,
                embedding_size=self.embedding_size,
                aggregation=self.aggregation,
                update_globals=True,
                decode_globals=True,
                decode_edges=False,  # No edge output, only global output (Q-value)
                decode_edge_dimension=1,
                decode_global_dimension=1,  # Outputs single Q-value
                normalization=self.normalization
            )
            return network(graph)
        
        self.critic_update_net = hk.without_apply_rng(hk.transform(critic_update_forward))
        
        # Optimizers
        self.actor_optimizer = optax.adam(learning_rate=actor_lr)
        self.critic_optimizer = optax.adam(learning_rate=critic_lr)
    
    def init(self, key: PRNGKey) -> DDPGState:
        """Initialize the DDPG agent using the same logic as HeatmapOptimizer"""
        # Create dummy graphs using the same logic as HeatmapOptimizer
        num_nodes = 10
        k = 7
        num_edges = num_nodes * k
        
        if self.dummy_observation is None:
            num_node_features = self.num_node_features
            num_edge_features = self.num_edge_features
            num_globals = self.num_global_features
        else:
            num_globals = jnp.concatenate([v for k,v in self.dummy_observation.graph.globals.items()], axis=-1).shape[-1]
            num_node_features = jnp.concatenate([v for k,v in self.dummy_observation.graph.nodes.items()], axis=-1).shape[-1]
            num_edge_features = jnp.concatenate([v for k,v in self.dummy_observation.graph.edges.items()], axis=-1).shape[-1]
            num_globals = num_globals + 11 + 1  # add the training step feature, budget feature
            num_edge_features = num_edge_features + 1 + 1 + 6  # add the momentum, grad, params features
        
        # Create senders and receivers with the correct number of edges
        senders = jnp.repeat(jnp.arange(num_nodes), num_nodes)
        receivers = senders.reshape(num_nodes, num_nodes).transpose().flatten()
        senders = senders[:num_edges]
        receivers = receivers[:num_edges]
        
        # Dummy graph for update networks (full features)
        dummy_graph_update = GraphsTuple(
            nodes=jnp.zeros((num_nodes, num_node_features)),
            senders=senders,
            receivers=receivers, 
            edges=jnp.zeros((num_edges, num_edge_features)),
            globals=jnp.zeros((1, num_globals)),
            n_node=jnp.array([num_nodes]),
            n_edge=jnp.array([num_edges])
        )
        
        # For critic update, edges include action concatenation: +1 feature for action
        critic_dummy_graph_update = dummy_graph_update._replace(
            edges=jnp.zeros((num_edges, num_edge_features + 1))
        )
        
        # Split the key for network initializations
        key_actor, key_critic_update = jax.random.split(key, 2)
        
        # Initialize actor parameters using HeatmapOptimizer
        actor_params = self.heatmap_optimizer.init(key_actor)
        
        # Initialize critic update network parameters (critic uses update net for all timesteps)
        critic_params = {
            'update_params': self.critic_update_net.init(key_critic_update, critic_dummy_graph_update)
        }
        
        # Create TrainStates with target parameters
        actor_state = DDPGTrainState.create(
            apply_fn=None,  # We'll use the networks directly
            params=actor_params,
            target_params=actor_params,
            tx=self.actor_optimizer
        )
        
        critic_state = DDPGTrainState.create(
            apply_fn=None,  # We'll use the networks directly
            params=critic_params,
            target_params=critic_params,
            tx=self.critic_optimizer
        )
        
        return DDPGState(
            actor_state=actor_state,
            critic_state=critic_state,
            step=0
        )
    
    def get_actor_params(self, state: Any) -> Any:
        """Return actor params in HeatmapOptimizer format from either object or dict state."""
        # Support dict checkpoints restored by Orbax as well as live DDPGState objects.
        actor_state = None
        if hasattr(state, 'actor_state'):
            actor_state = state.actor_state
        elif isinstance(state, dict) and 'actor_state' in state:
            actor_state = state['actor_state']

        if actor_state is None:
            # If a raw params pytree was passed in, just return it
            return state

        if hasattr(actor_state, 'params'):
            return actor_state.params
        elif isinstance(actor_state, dict) and 'params' in actor_state:
            return actor_state['params']
        else:
            return actor_state
    
    
    @partial(jax.jit, static_argnums=0)
    def update_critic(self,
                     actor_state: DDPGTrainState,
                     critic_state: DDPGTrainState,
                     states: Any,
                     actions: Any,
                     next_states: Any,
                     rewards: Any,
                     dones: Any,
                     timesteps: Any = None):
        """Update critic network using target networks.
        
        Uses init_net for t=0, update_net for t>0 to get actions.
        Critic always uses update_net (zero-padded for t=0).
        """
        # Get next actions from target actor
        # next_states are time-shifted, so they're all from t>=1 and have optimizer features
        # Therefore, always use update_net (pass timesteps=None to avoid init_net)
        next_actions = self._get_actions(actor_state.target_params, next_states, timesteps=None)
        
        # Concatenate actions to states for critic input
        states_with_actions = self._concatenate_action_to_state(states, actions)
        next_states_with_actions = self._concatenate_action_to_state(next_states, next_actions)
        
        # Compute target Q-values using critic update network
        # next_states are time-shifted, so timesteps for next_states are timesteps+1
        # But we pass None since they're all t>=1 and don't need zero-padding
        next_q_values = self._compute_q_values(critic_state.target_params, next_states_with_actions, timesteps=None)
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        def mse_loss(params):
            # Current states already contain correct t=0 zero features from lopt
            current_q_values = self._compute_q_values(params, states_with_actions, timesteps=None)
            return ((current_q_values - target_q_values) ** 2).mean(), current_q_values.mean()
        
        (critic_loss, q_values), grads = jax.value_and_grad(mse_loss, has_aux=True)(critic_state.params)
        new_critic_state = critic_state.apply_gradients(grads=grads)
        
        return new_critic_state, critic_loss, q_values
    
    @partial(jax.jit, static_argnums=0)
    def update_actor(self,
                    actor_state: DDPGTrainState,
                    critic_state: DDPGTrainState,
                    states: Any,
                    timesteps: Any = None):
        """Update actor network using critic evaluation.
        
        Uses init_net for t=0, update_net for t>0 to get actions.
        Critic uses update_net for all timesteps (zero-padded for t=0).
        """
        def actor_loss(params):
            # Get actions using appropriate network (init_net for t=0, update_net for t>0)
            actions = self._get_actions(params, states, timesteps)
            states_with_actions = self._concatenate_action_to_state(states, actions)
            # States already have proper t=0 zeros from lopt; no padding needed
            q_values = self._compute_q_values(critic_state.params, states_with_actions, timesteps=None)
            return -q_values.mean()
        
        actor_loss_value, grads = jax.value_and_grad(actor_loss)(actor_state.params)
        new_actor_state = actor_state.apply_gradients(grads=grads)
        
        # Update target networks
        new_actor_state = new_actor_state.replace(
            target_params=optax.incremental_update(new_actor_state.params, new_actor_state.target_params, self.tau)
        )
        
        new_critic_state = critic_state.replace(
            target_params=optax.incremental_update(critic_state.params, critic_state.target_params, self.tau)
        )
        
        return new_actor_state, new_critic_state, actor_loss_value
    
    def _get_actions(self, actor_params: Any, states: Any, timesteps: Any = None) -> Any:
        """Get actions from HeatmapOptimizer networks.
        
        Uses init_net for t=0 states (no optimizer features yet),
        and update_net for t>0 states (with optimizer features).
        """
        # Use update network for all timesteps (t=0 has zero-padded optimizer features)
        output = self.heatmap_optimizer.update_net.apply(actor_params['update_params'], states)
        return output.edges
    
    def _compute_q_values(self, critic_params: Any, states_with_actions: Any, timesteps: Any = None) -> Any:
        """Compute Q-values using the critic update network.
        
        Uses only the update network for all timesteps. States from lopt already
        have t=0 optimizer features set to zero.
        
        Args:
            critic_params: Critic network parameters
            states_with_actions: GraphsTuple with actions concatenated to edges
            timesteps: Unused (kept for signature compatibility)
        
        Returns:
            Q-values from the critic network globals (shape: (batch_size, 1))
        """
        return self.critic_update_net.apply(critic_params['update_params'], states_with_actions).globals
    
    def _concatenate_action_to_state(self, states: Any, actions: Any) -> Any:
        """Concatenate action (heatmap) to edge features of the graph state"""
        
        # Concatenate action to edge features
        new_edge_features = jnp.concatenate([states.edges, actions], axis=-1)
        
        # Create new GraphsTuple with updated edge features
        new_state = states._replace(edges=new_edge_features)
        
        return new_state
    
    def update(self, state: DDPGState, batch: Any, train_actor: bool = True, train_critic: bool = True) -> Tuple[DDPGState, Any]:
        """Update DDPG agent using separate critic and actor updates"""
        # batch contains: states, actions, rewards, next_states, dones, timesteps
        if len(batch) == 6:
            states, actions, rewards, next_states, dones, timesteps = batch
        else:
            # Backward compatibility if timesteps not provided
            states, actions, rewards, next_states, dones = batch
            timesteps = None
        
        # # Update critic
        # new_critic_state, critic_loss, q_values = self.update_critic(
        #     state.actor_state,
        #     state.critic_state,
        #     states,
        #     actions,
        #     next_states,
        #     rewards,
        #     dones,
        #     timesteps
        # )
        
        # # Update actor
        # new_actor_state, updated_critic_state, actor_loss = self.update_actor(
        #     state.actor_state,
        #     new_critic_state,
        #     states,
        #     timesteps
        # )
        
        # # Create new state
        # new_state = state.replace(
        #     actor_state=new_actor_state,
        #     critic_state=updated_critic_state,
        #     step=state.step + 1
        # )
        
        # metrics = {
        #     'critic_loss': critic_loss,
        #     'actor_loss': actor_loss,
        #     'q_values': q_values
        # }
        # metrics = {k: float(jnp.asarray(jax.device_get(v)).reshape(-1)[0]) for k, v in metrics.items()} 
        # return new_state, metrics
        
        actor_state = state.actor_state
        critic_state = state.critic_state
        critic_loss = jnp.array(0.0)
        q_values = jnp.array(0.0)

        # 1) Critic update (always uses target actor/critic)
        if train_critic:
            critic_state, critic_loss, q_values = self.update_critic(
                actor_state,
                critic_state,
                states,
                actions,
                next_states,
                rewards,
                dones,
                timesteps
            )

        # 2) Actor update (+ Polyak of targets) only if requested
        actor_loss = jnp.array(0.0)
        if train_actor:
            actor_state, critic_state, actor_loss = self.update_actor(
                actor_state,
                critic_state,
                states,
                timesteps
            )
            # NOTE: update_actor already Polyak-updates both targets.
            # If train_actor=False, we do NOT Polyak targets (matching CleanRL).

        new_state = state.replace(
            actor_state=actor_state,
            critic_state=critic_state,
            step=state.step + 1
        )

        metrics = {
            'critic_loss': critic_loss,
            'actor_loss': actor_loss,
            'q_values': q_values
        }
        # make them plain floats for logging
        metrics = {k: float(jnp.asarray(v).reshape(())) for k, v in metrics.items()}
        return new_state, metrics
