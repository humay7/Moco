# TSP learnable optimizer
import jax 
import jax.numpy as jnp
from jax import lax
import chex
from chex import Array, ArrayTree, PRNGKey
import jraph
from jraph import GraphsTuple, GraphNetwork

import haiku as hk

import flax
from typing import Any, Callable, Optional, Tuple, Union

from learned_optimization.learned_optimizers import base as lopt_base
from learned_optimization.optimizers import base as opt_base
from learned_optimization.learned_optimizers import common
from learned_optimization import tree_utils
from learned_optimization import summary

from moco.gnn import GNN
from moco.gnn2 import GCN

MetaParams = Any
LOptState = Any

def _second_moment_normalizer(x, axis, eps=1e-5):
  return x * lax.rsqrt(eps + jnp.mean(jnp.square(x), axis=axis, keepdims=True))


def _tanh_embedding(iterations):
  f32 = jnp.float32

  def one_freq(timescale):
    return jnp.tanh(iterations / (f32(timescale)) - 1.0)

  timescales = jnp.asarray(
      [1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000],
      dtype=jnp.float32)
  return jax.vmap(one_freq)(timescales)

@flax.struct.dataclass
class GNNLOptState:
  params: Any
  rolling_features: common.MomAccumulator
  state: Any # model state, required by learned optimization interface
  iteration: jnp.ndarray
  budget: jnp.ndarray
  augmented_graph: Any = None  # Store the augmented graph with optimizer features
  stats: Any = None  # Optional per-step debugging stats

import haiku as hk

class HeatmapOptimizer(lopt_base.LearnedOptimizer):
  def __init__(self, update_strategy='direct', embedding_size=64, num_layers_init=3, num_layers_update=3, aggregation='max', normalize_inputs=False, compute_summary=True, num_node_features=1, num_edge_features=41, num_global_features=45, normalization="pre", dummy_observation=None):
    self._compute_summary = compute_summary
    self.embedding_size = embedding_size
    self.num_layers_update = num_layers_update
    self.num_layers_init = num_layers_init
    self.normalize_inputs = normalize_inputs
    self.num_node_features = num_node_features
    self.num_edge_features = num_edge_features
    self.num_global_features = num_global_features
    self.dummy_observation = dummy_observation
    # self.total_steps = total_steps
    self.aggregation = aggregation
    self.normalization = normalization
    # update strategy describes whether the optimizer should put out the new heatmap directly, put out the new heatmap with seperate temperature or only put out the difference to the old heatmap 
    self.update_strategy = update_strategy # options: 'direct', 'temperature', 'difference'

    def update_forward(graph):
      network = GNN(
        num_layers = self.num_layers_update, 
        embedding_size= self.embedding_size,
        aggregation = self.aggregation,
        update_globals = True, 
        decode_globals = True if self.update_strategy == 'temperature' else False, 
        decode_edges = True, 
        decode_edge_dimension = 1, 
        decode_global_dimension = 1,
        normalization=self.normalization
        )
      return network(graph)
    
    self.update_net = hk.without_apply_rng(hk.transform(update_forward))

  def init(self, key) -> MetaParams:
    """Initialize the weights of the learned optimizer."""
    # we create a dummy graph since the gnn can operate on different sized graphs 
    num_nodes = 10
    k = 7
    num_edges = num_nodes * k
    if self.dummy_observation is None:
      num_node_features = self.num_node_features # dummy feature
      num_edge_features = self.num_edge_features # params, grad, momentum (6), dists, best_global (top_k),
      num_globals = self.num_global_features # best_cost, mean_cost_batch, step tanh_embedding (11), topk_gaps (top_k), rel_impr, step/budget
    else:
      num_globals = jnp.concatenate([v for k,v in self.dummy_observation.graph.globals.items()], axis=-1).shape[-1]
      num_node_features = jnp.concatenate([v for k,v in self.dummy_observation.graph.nodes.items()], axis=-1).shape[-1]
      num_edge_features = jnp.concatenate([v for k,v in self.dummy_observation.graph.edges.items()], axis=-1).shape[-1]
      num_globals = num_globals + 11 + 1# add the training step feature, budget feature
      num_edge_features = num_edge_features + 1 + 1 + 6 # add the momentum, grad, params features

    # create senders and receivers with the correct number of edges
    senders = jnp.repeat(jnp.arange(num_nodes), num_nodes)
    receivers = senders.reshape(num_nodes, num_nodes).transpose().flatten()
    senders = senders[:num_edges]
    receivers = receivers[:num_edges]

    dummy_graph_update = GraphsTuple(
        nodes= jnp.zeros((num_nodes, num_node_features)),
        senders= senders,
        receivers= receivers, 
        edges= jnp.zeros((num_edges, num_edge_features)),
        globals= jnp.zeros((1, num_globals)),
        n_node= jnp.array([num_nodes]),
        n_edge= jnp.array([num_edges])
    )

    params = {
      'update_params': self.update_net.init(key, dummy_graph_update)
    }
    return params

  def opt_fn(self, theta: MetaParams, is_training: bool = False) -> opt_base.Optimizer:
    # define an anonymous class which implements the optimizer.
    # this captures over the meta-parameters, theta.
    decays = jnp.asarray([0.1, 0.5, 0.9, 0.99, 0.999, 0.9999])
    normalize = self.normalize_inputs
    update_net = self.update_net
    compute_summary = self._compute_summary
    update_strategy = self.update_strategy
    # total_steps = self.total_steps

    class _Opt(opt_base.Optimizer):
      def init(self,
               params: lopt_base.Params,
               model_state: Any = None,
               num_steps: Optional[int] = None,
               key: Optional[PRNGKey] = None) -> GNNLOptState:
        """Initialize inner opt state."""
        n,k = params['params']['heatmap'].shape

        # Build an initial augmented graph with consistent structure/types used in updates
        # Edge optimizer features: zeros for momentum(6), grad(1), params(1) at t=0
        e = n * k
        momentum0 = jnp.zeros((e, 6))
        grad0 = jnp.zeros((e, 1))
        param_feat0 = jnp.zeros((e, 1))
        stacked_inp0 = jnp.concatenate([momentum0, grad0, param_feat0], axis=-1)  # (e,8)

        edge_feats0 = [v for _, v in model_state.graph.edges.items()]
        stacked_edge0 = jnp.concatenate(edge_feats0 + [stacked_inp0], axis=-1)

        node_feats0 = [v for _, v in model_state.graph.nodes.items()]
        stacked_node0 = jnp.concatenate(node_feats0, axis=-1)

        global_feats0 = [v for _, v in model_state.graph.globals.items()]
        training_step_feature0 = _tanh_embedding(jnp.asarray(0, dtype=jnp.int32)).reshape(1, -1)
        budget0 = jnp.asarray(num_steps if num_steps is not None else 1, dtype=jnp.int32)
        budget_feature0 = (jnp.asarray(0, dtype=jnp.int32) / budget0).reshape(1, 1)
        stacked_global0 = jnp.concatenate(global_feats0 + [training_step_feature0, budget_feature0], axis=-1)

        augmented_graph0 = GraphsTuple(
          nodes=stacked_node0,
          edges=stacked_edge0,
          globals=stacked_global0,
          senders=model_state.graph.senders,
          receivers=model_state.graph.receivers,
          n_node=model_state.graph.n_node,
          n_edge=model_state.graph.n_edge,
        )

        # Initialize heatmap by running update_net on t=0 augmented graph
        output0 = update_net.apply(theta['update_params'], augmented_graph0)
        # Build initial stats with same structure as in update()
        def _stats(x, prefix):
          return {
            f'{prefix}_nan': jnp.any(jnp.isnan(x)),
            f'{prefix}_inf': jnp.any(jnp.isinf(x)),
            f'{prefix}_min': jnp.nanmin(x),
            f'{prefix}_max': jnp.nanmax(x),
            f'{prefix}_mean': jnp.nanmean(x),
            f'{prefix}_std': jnp.nanstd(x),
          }
        init_stats = {}
        init_stats.update(_stats(stacked_global0, 'pre_apply_globals'))
        init_stats.update(_stats(stacked_edge0, 'pre_apply_edges'))
        init_stats.update(_stats(stacked_node0, 'pre_apply_nodes'))
        init_stats.update(_stats(output0.edges, 'post_apply_output_edges'))
        if update_strategy == 'temperature' and (output0.globals is not None):
          temp0 = 0.5 * (jnp.tanh(output0.globals) + 1.0)
          init_stats.update(_stats(output0.globals, 'post_apply_output_globals'))
          init_stats.update(_stats(temp0, 'post_apply_temp'))
        else:
          # If no globals decoded, still provide consistent keys
          if hasattr(output0, 'globals') and (output0.globals is not None):
            init_stats.update(_stats(output0.globals, 'post_apply_output_globals'))
          else:
            # create placeholder zeros for globals stats
            z = jnp.array(0.0)
            init_stats.update({'post_apply_output_globals_nan': jnp.array(False),
                               'post_apply_output_globals_inf': jnp.array(False),
                               'post_apply_output_globals_min': z,
                               'post_apply_output_globals_max': z,
                               'post_apply_output_globals_mean': z,
                               'post_apply_output_globals_std': z})
          init_stats.update({'post_apply_temp_nan': jnp.array(False), 'post_apply_temp_inf': jnp.array(False),
                             'post_apply_temp_min': jnp.array(1.0), 'post_apply_temp_max': jnp.array(1.0),
                             'post_apply_temp_mean': jnp.array(1.0), 'post_apply_temp_std': jnp.array(0.0)})
        params = flax.core.unfreeze(params)
        params['params']['heatmap'] = output0.edges.reshape(n, k)
        params = flax.core.freeze(params)

        return GNNLOptState(
            params=params,
            state=model_state,
            rolling_features=common.vec_rolling_mom(decays).init(params),
            iteration=jnp.asarray(0, dtype=jnp.int32),
            budget=jnp.asarray(num_steps, dtype=jnp.int32),
            augmented_graph=augmented_graph0,
            stats=init_stats
        )
      def update(
          self,  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
          opt_state: GNNLOptState,
          grad: Any,
          loss: float,
          model_state: Any = None,
          is_valid: bool = False,
          key: Optional[PRNGKey] = None,

      ) -> GNNLOptState:
        # remove cost for now
        # do not normalize features
        n, k = opt_state.params['params']['heatmap'].shape
        e = n * k
        # compute momentum, and other additional features
        next_rolling_features = common.vec_rolling_mom(decays).update(
            opt_state.rolling_features, grad)
        momentum = next_rolling_features.m['params']['heatmap'].reshape(e,-1)
        params = opt_state.params['params']['heatmap'].reshape(e,-1)
        # check if the reshape is correct and doesnt break the ordering of the edges
        # jnp.arange(20*18*6).reshape(20,18,6).reshape(20*18,6) # seems to be correct

        training_step_feature = _tanh_embedding(opt_state.iteration).reshape(1, -1)
        budget_feature = (opt_state.iteration / opt_state.budget).reshape(1,1)
        
        # construct graph
        global_feats = [v for k,v in model_state.graph.globals.items()]
        node_feats = [v for k,v in model_state.graph.nodes.items()]
        edge_feats = [v for k,v in model_state.graph.edges.items()]

        additional_inp = [momentum, grad['params']['heatmap'].reshape(-1,1), params]
        stacked_inp = jnp.concatenate(additional_inp, axis=-1)
        if normalize:
          stacked_inp = _second_moment_normalizer(stacked_inp, axis=list(range(len(stacked_inp.shape))))
        edge_feats.append(stacked_inp)

        # stack
        stacked_global = jnp.concatenate(global_feats, axis=-1)
        stacked_node = jnp.concatenate(node_feats, axis=-1)
        stacked_edge = jnp.concatenate(edge_feats, axis=-1)

        # once normalized, add features that are constant.
        stacked_global = jnp.concatenate([stacked_global, training_step_feature, budget_feature], axis=-1)

        graph = GraphsTuple(
            nodes=stacked_node,
            edges=stacked_edge,
            globals=stacked_global,
            senders=model_state.graph.senders,
            receivers=model_state.graph.receivers,
            n_node=model_state.graph.n_node,
            n_edge=model_state.graph.n_edge
        )

        # apply GNN
        output = update_net.apply(theta['update_params'], graph)

        # Collect compact stats for outer logging
        def _stats(x, prefix):
          return {
            f'{prefix}_nan': jnp.any(jnp.isnan(x)),
            f'{prefix}_inf': jnp.any(jnp.isinf(x)),
            f'{prefix}_min': jnp.nanmin(x),
            f'{prefix}_max': jnp.nanmax(x),
            f'{prefix}_mean': jnp.nanmean(x),
            f'{prefix}_std': jnp.nanstd(x),
          }
        step_stats = {}
        step_stats.update(_stats(stacked_global, 'pre_apply_globals'))
        step_stats.update(_stats(stacked_edge, 'pre_apply_edges'))
        step_stats.update(_stats(stacked_node, 'pre_apply_nodes'))
        step_stats.update(_stats(output.edges, 'post_apply_output_edges'))
        step_stats.update(_stats(output.globals, 'post_apply_output_globals'))

        if update_strategy == 'temperature':
          # Temperature in (0,1); clamp away from 0/1 to avoid infinities and flat gradients
          temp_raw = 0.5 * (jax.nn.tanh(output.globals) + 1.0)
          tmin = jnp.asarray(1e-2, dtype=output.globals.dtype)
          tmax = jnp.asarray(1.0 - 1e-6, dtype=output.globals.dtype)
          temp = jnp.clip(temp_raw, tmin, tmax)
          new_p = (output.edges / temp).reshape(n, k)
          step_stats.update(_stats(temp, 'post_apply_temp'))
        else:
          # direct (no temperature) or other strategies: ensure temp-* keys present for consistent pytree
          new_p = output.edges.reshape(n, k)
          step_stats.update({
            'post_apply_temp_nan': jnp.array(False),
            'post_apply_temp_inf': jnp.array(False),
            'post_apply_temp_min': jnp.array(1.0),
            'post_apply_temp_max': jnp.array(1.0),
            'post_apply_temp_mean': jnp.array(1.0),
            'post_apply_temp_std': jnp.array(0.0),
          })

        new_opt_state = GNNLOptState(
          params=jax.tree_util.tree_map(lambda _: new_p, opt_state.params),
          state=model_state,
          rolling_features=tree_utils.match_type(next_rolling_features, opt_state.rolling_features),
          iteration=opt_state.iteration + 1,
          budget=opt_state.budget,
          augmented_graph=graph,  # Store the augmented graph
          stats=step_stats,
        )
        return new_opt_state
    return _Opt()
  

class MisOptimizer(lopt_base.LearnedOptimizer):
  def __init__(self, embedding_size=64, num_layers_init=3, num_layers_update=3, aggregation='max', compute_summary=True, num_node_features=41, num_global_features=45, dummy_observation=None):
    self._compute_summary = compute_summary
    self.embedding_size = embedding_size
    self.num_layers_update = num_layers_update
    self.num_layers_init = num_layers_init
    self.num_node_features = num_node_features
    self.num_global_features = num_global_features
    self.dummy_observation = dummy_observation
    self.aggregation = aggregation
    # self.normalization = normalization

    def update_forward(graph):
      network = GCN(
        num_layers = self.num_layers_update, 
        embedding_size = self.embedding_size,
        aggregation = self.aggregation,
        embed_globals = True,
        update_globals = True, 
        decode_globals = True,
        decode_node_dimension = 1, 
        decode_global_dimension = 1,
        # normalization = self.normalization
        )
      return network(graph)
    
    def init_forward(graph):
      network = GCN(
        num_layers = self.num_layers_update, 
        embedding_size = self.embedding_size,
        aggregation = self.aggregation,
        embed_globals = False,
        update_globals = False, 
        decode_globals = False,
        decode_node_dimension = 1, 
        decode_global_dimension = 1,
        # normalization = self.normalization
        )
      return network(graph)

    self.init_net = hk.without_apply_rng(hk.transform(init_forward))
    self.update_net = hk.without_apply_rng(hk.transform(update_forward))

  def init(self, key) -> MetaParams:
    """Initialize the weights of the learned optimizer."""
    # we create a dummy graph since the gnn can operate on different sized graphs 
    num_nodes = num_edges = 3 
    if self.dummy_observation is None:
      num_node_features = self.num_node_features # dummy feature  
      # num_edge_features = self.num_edge_features # params, grad, momentum (6), dists, best_global (top_k),
      num_globals = self.num_global_features # best_cost, mean_cost_batch, step tanh_embedding (11), topk_gaps (top_k), rel_impr, step/budget
    else:
      num_globals = jnp.concatenate([v for k,v in self.dummy_observation.graph.globals.items()], axis=-1).shape[-1]
      num_node_features = jnp.concatenate([v for k,v in self.dummy_observation.graph.nodes.items()], axis=-1).shape[-1]
      num_globals = num_globals + 11 + 1 # add the training step feature, budget feature
      num_node_features = num_node_features + 1 + 1 + 6 # add the momentum, grad, params features

    # create senders and receivers with the correct number of edges
    senders = jnp.arange(num_edges)
    receivers = senders

    dummy_graph_update = GraphsTuple(
        nodes = jnp.zeros((num_nodes, num_node_features)),
        senders = senders,
        receivers = receivers, 
        edges = None,
        globals = jnp.zeros((1, num_globals)),
        n_node = jnp.array([num_nodes]),
        n_edge = jnp.array([num_edges])
    )

    dummy_graph_init = GraphsTuple(
        nodes= jnp.zeros((num_nodes, 1)),
        globals = None,
        senders= senders,
        receivers= receivers, 
        edges= None,
        n_node= jnp.array([num_nodes]),
        n_edge= jnp.array([num_edges])
    )

    params = {
      'init_params': self.init_net.init(key, dummy_graph_init),
      'update_params': self.update_net.init(key, dummy_graph_update)
    }
    return params

  def opt_fn(self, theta: MetaParams, is_training: bool = False) -> opt_base.Optimizer:
    # define an anonymous class which implements the optimizer.
    # this captures over the meta-parameters, theta.
    decays = jnp.asarray([0.1, 0.5, 0.9, 0.99, 0.999, 0.9999])
    init_net = self.init_net
    update_net = self.update_net

    class _Opt(opt_base.Optimizer):
      def init(self,
               params: lopt_base.Params,
               model_state: Any = None,
               num_steps: Optional[int] = None,
               key: Optional[PRNGKey] = None) -> GNNLOptState:
        """Initialize inner opt state."""

        # initialize the heatmap from model
        graph = model_state.graph._replace(
          edges = None,
          globals = None,
          nodes = model_state.graph.nodes['dummy'])

        output = init_net.apply(theta['init_params'], graph)

        params = flax.core.unfreeze(params)
        params['params']['heatmap'] = output.nodes.squeeze()
        params = flax.core.freeze(params)

        return GNNLOptState(
            params=params,
            state=model_state,
            rolling_features=common.vec_rolling_mom(decays).init(params),
            iteration=jnp.asarray(0, dtype=jnp.int32),
            budget=jnp.asarray(num_steps, dtype=jnp.int32),
            augmented_graph=None  # No augmented graph at initialization
        )
      def update(
          self,  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
          opt_state: GNNLOptState,
          grad: Any,
          loss: float,
          model_state: Any = None,
          is_valid: bool = False,
          key: Optional[PRNGKey] = None,

      ) -> GNNLOptState:
        # compute momentum, and other additional features
        next_rolling_features = common.vec_rolling_mom(decays).update(
            opt_state.rolling_features, grad)
        momentum = next_rolling_features.m['params']['heatmap']
        params = opt_state.params['params']['heatmap']

        training_step_feature = _tanh_embedding(opt_state.iteration).reshape(1, -1)
        budget_feature = (opt_state.iteration / opt_state.budget).reshape(1,1)
        
        # construct graph
        global_feats = [v for k,v in model_state.graph.globals.items()]
        # repeat is necessary because graph is padded with second graph
        global_feats += [jnp.repeat(training_step_feature, 2, axis=0, total_repeat_length=2), 
                         jnp.repeat(budget_feature, 2, axis=0, total_repeat_length=2)]
        node_feats = [v for k,v in model_state.graph.nodes.items()]
        node_feats += [momentum, grad['params']['heatmap'].reshape(-1,1), params.reshape(-1,1)]

        # stack
        stacked_global = jnp.concatenate(global_feats, axis=-1)
        stacked_node = jnp.concatenate(node_feats, axis=-1)

        graph = model_state.graph._replace(
            nodes=stacked_node,
            globals=stacked_global,
        )

        # apply GNN
        output = update_net.apply(theta['update_params'], graph)
        # Temperature in (0,1); clamp away from 0 to avoid blow-ups when dividing
        temp_raw = 0.5 * (jax.nn.tanh(output.globals) + 1.0)
        tmin = jnp.asarray(1e-2, dtype=output.globals.dtype)
        tmax = jnp.asarray(1.0 - 1e-6, dtype=output.globals.dtype)
        temp = jnp.clip(temp_raw, tmin, tmax)[0]
        new_p = (output.nodes / temp).squeeze()

        new_opt_state = GNNLOptState(
          params=jax.tree_util.tree_map(lambda _: new_p, opt_state.params),
          state=model_state,
          rolling_features=tree_utils.match_type(next_rolling_features, opt_state.rolling_features),
          iteration=opt_state.iteration + 1,
          budget=opt_state.budget,
          augmented_graph=graph  # Store the augmented graph
        )
        return new_opt_state
    return _Opt()
