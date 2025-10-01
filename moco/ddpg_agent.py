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

@struct.dataclass
class DDPGTrainState(TrainState):
    """TrainState with target parameters for DDPG"""
    target_params: FrozenDict

from moco.gnn import GNN

@struct.dataclass
class DDPGState:
    """State for DDPG training"""
    actor_state: DDPGTrainState
    critic_state: DDPGTrainState
    step: int

class DDPGAgent:
    """DDPG agent for learning phi_init and phi_update parameters"""
    
    def __init__(self, 
                 update_strategy='direct',
                 embedding_size=64, 
                 num_layers_init=3, 
                 num_layers_update=3, 
                 aggregation='max', 
                 normalize_inputs=False, 
                 compute_summary=True, 
                 num_node_features=1, 
                 num_edge_features=41, 
                 num_global_features=45, 
                 normalization="pre", 
                 dummy_observation=None,
                 actor_lr: float = 1e-4,
                 critic_lr: float = 1e-3,
                 tau: float = 0.005,
                 gamma: float = 0.99):
        
        self._compute_summary = compute_summary
        self.embedding_size = embedding_size
        self.num_layers_update = num_layers_update
        self.num_layers_init = num_layers_init
        self.normalize_inputs = normalize_inputs
        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features
        self.num_global_features = num_global_features
        self.dummy_observation = dummy_observation
        self.aggregation = aggregation
        self.normalization = normalization
        self.update_strategy = update_strategy
        
        # DDPG specific parameters
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.tau = tau
        self.gamma = gamma
        
        # Actor network - same structure as HeatmapOptimizer's update net only
        def update_forward(graph):
            network = GNN(
                num_layers=self.num_layers_update, 
                embedding_size=self.embedding_size,
                aggregation=self.aggregation,
                update_globals=True, 
                decode_globals=True if self.update_strategy == 'temperature' else False, 
                decode_edges=True, 
                decode_edge_dimension=1, 
                decode_global_dimension=1,
                normalization=self.normalization
            )
            return network(graph)
        
        self.update_net = hk.without_apply_rng(hk.transform(update_forward))
        
        # Critic network - same GNN structure but decode_globals=True, decode_edges=False
        # Action gets concatenated to edge features
        def critic_forward(graph):
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
        
        self.critic_net = hk.without_apply_rng(hk.transform(critic_forward))
        
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

        # Persist inferred feature sizes for runtime padding/concatenation
        self.num_node_features = int(num_node_features)
        self.num_edge_features = int(num_edge_features)
        self.num_global_features = int(num_globals)
        
        # Create senders and receivers with the correct number of edges
        senders = jnp.repeat(jnp.arange(num_nodes), num_nodes)
        receivers = senders.reshape(num_nodes, num_nodes).transpose().flatten()
        senders = senders[:num_edges]
        receivers = receivers[:num_edges]
        
        dummy_graph_update = GraphsTuple(
            nodes=jnp.zeros((num_nodes, num_node_features)),
            senders=senders,
            receivers=receivers, 
            edges=jnp.zeros((num_edges, num_edge_features)),
            globals=jnp.zeros((1, num_globals)),
            n_node=jnp.array([num_nodes]),
            n_edge=jnp.array([num_edges])
        )
        
        # For critic, edges include action concatenation: +1 feature for action
        critic_dummy_graph_update = dummy_graph_update._replace(
            edges=jnp.zeros((num_edges, num_edge_features + 1))
        )
        
        # Initialize actor network (only update net)
        actor_params = self.update_net.init(key, dummy_graph_update)
        
        # Initialize critic network with state+action edge feature size
        critic_params = self.critic_net.init(key, critic_dummy_graph_update)
        
        # Create TrainStates with target parameters
        actor_state = DDPGTrainState.create(
            apply_fn=self.update_net.apply,
            params=actor_params,
            target_params=actor_params,
            tx=self.actor_optimizer
        )
        
        critic_state = DDPGTrainState.create(
            apply_fn=self.critic_net.apply,
            params=critic_params,
            target_params=critic_params,
            tx=self.critic_optimizer
        )
        
        return DDPGState(
            actor_state=actor_state,
            critic_state=critic_state,
            step=0
        )
    
    def get_actor_params(self, state: DDPGState) -> Any:
        """Get current actor parameters in the same format as HeatmapOptimizer"""
        # HeatmapOptimizer now uses parameters directly (no dictionary)
        return state.actor_state.params
    
    
    @partial(jax.jit, static_argnums=0)
    def update_critic(self,
                     actor_state: DDPGTrainState,
                     critic_state: DDPGTrainState,
                     states: Any,
                     actions: Any,
                     next_states: Any,
                     rewards: Any,
                     dones: Any):
        """Update critic network using target networks"""
        # Get next actions from target actor
        next_actions = self._get_actions(actor_state.target_params, next_states)
        
        # Concatenate actions to states for critic input
        states_with_actions = self._concatenate_action_to_state(states, actions)
        next_states_with_actions = self._concatenate_action_to_state(next_states, next_actions)
        
        # Compute target Q-values
        next_q_values = self.critic_net.apply(critic_state.target_params, next_states_with_actions)
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        def mse_loss(params):
            current_q_values = self.critic_net.apply(params, states_with_actions)
            return ((current_q_values - target_q_values) ** 2).mean(), current_q_values.mean()
        
        (critic_loss, q_values), grads = jax.value_and_grad(mse_loss, has_aux=True)(critic_state.params)
        new_critic_state = critic_state.apply_gradients(grads=grads)
        
        return new_critic_state, critic_loss, q_values
    
    @partial(jax.jit, static_argnums=0)
    def update_actor(self,
                    actor_state: DDPGTrainState,
                    critic_state: DDPGTrainState,
                    states: Any):
        """Update actor network"""
        def actor_loss(params):
            actions = self._get_actions(params, states)
            states_with_actions = self._concatenate_action_to_state(states, actions)
            q_values = self.critic_net.apply(critic_state.params, states_with_actions)
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
    
    def _get_actions(self, actor_params: Any, states: Any) -> Any:
        """Get actions from actor networks"""
        # Use the update network to generate actions (heatmaps); return edge outputs
        output = self.update_net.apply(actor_params, states)
        return output.edges
    
    def _concatenate_action_to_state(self, states: Any, actions: Any) -> Any:
        """Concatenate action (heatmap) to edge features of the graph state"""
        
        # Concatenate action to edge features
        new_edge_features = jnp.concatenate([states.edges, actions], axis=-1)
        
        # Create new GraphsTuple with updated edge features
        new_state = states._replace(edges=new_edge_features)
        
        return new_state
    
    def update(self, state: DDPGState, batch: Any) -> Tuple[DDPGState, Any]:
        """Update DDPG agent using separate critic and actor updates"""
        # batch contains: states, actions, rewards, next_states, dones
        states, actions, rewards, next_states, dones = batch
        
        # Update critic
        new_critic_state, critic_loss, q_values = self.update_critic(
            state.actor_state,
            state.critic_state,
            states,
            actions,
            next_states,
            rewards,
            dones
        )
        
        # Update actor
        new_actor_state, updated_critic_state, actor_loss = self.update_actor(
            state.actor_state,
            new_critic_state,
            states
        )
        
        # Create new state
        new_state = state.replace(
            actor_state=new_actor_state,
            critic_state=updated_critic_state,
            step=state.step + 1
        )
        
        metrics = {
            'critic_loss': critic_loss,
            'actor_loss': actor_loss,
            'q_values': q_values
        }
        
        return new_state, metrics
