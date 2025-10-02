# GNN in Jax
import jax
import jax.numpy as jnp

import chex
from chex import Array
import jraph
from jraph import GraphsTuple, GraphNetwork
from jraph._src.utils import segment_sum, segment_max
import haiku as hk
# import jmp

class GNN(hk.Module):

  def __init__(self, 
    num_layers: int = 5,
    embedding_size: int = 64,
    aggregation = 'max',
    embed_globals = True,
    update_globals = True,
    decode_globals = False,
    decode_edges = True,
    decode_edge_dimension = 1,
    decode_global_dimension: int = 1,
    normalization = 'none', # pre, post, none
    name="GNN"):

    super().__init__(name=name)
    self.num_layers = num_layers
    self.embedding_size = embedding_size
    self.aggregation = aggregation
    self.embed_globals = embed_globals
    self.update_globals = update_globals
    self.decode_globals = decode_globals
    self.decode_edges = decode_edges
    self.decode_edge_dimension = decode_edge_dimension
    self.decode_global_dimension = decode_global_dimension
    self.normalization = normalization

    if self.aggregation == 'max':
       self.aggregate_fn = segment_max
    elif self.aggregation == 'sum':
       self.aggregate_fn = segment_sum

  # -------------------- shape helpers --------------------

  @staticmethod
  def _squeeze_extra_unit_axes(x, target_ndim):
    """If x has rank > target_ndim, squeeze unit axes (from the left) until ranks match."""
    while x.ndim > target_ndim:
      # try to squeeze a leading unit axis; if not unit, we cannot safely drop it
      if x.shape[0] == 1:
        x = jnp.squeeze(x, axis=0)
      else:
        # try squeeze next leading axis if it is unit
        squeezed = False
        for ax in range(x.ndim - target_ndim):
          if x.shape[ax] == 1:
            x = jnp.squeeze(x, axis=ax)
            squeezed = True
            break
        if not squeezed:
          # can't safely squeeze: leave as-is and let broadcast checker fail later
          break
    return x

  @staticmethod
  def _broadcast_to_ref(x, ref, name: str):
    """
    Make `x` align with `ref` on all leading axes (all except last feat axis).
    Steps:
      1) If x has higher rank than ref, squeeze only unit axes to at most ref.ndim.
      2) If x has lower rank, insert singleton axes before the last axis until ranks match.
      3) Broadcast to ref.shape[:-1] + (x_feat,).
    """
    if x is None:
      return None

    # 1) reduce rank if needed (only unit axes)
    if x.ndim > ref.ndim:
      x = GNN._squeeze_extra_unit_axes(x, ref.ndim)

    # 2) insert singleton axes to match rank
    while x.ndim < ref.ndim:
      x = x[..., None, :]  # add a singleton axis before the feature axis

    # 3) broadcast leading dims to ref
    target_shape = ref.shape[:-1] + (x.shape[-1],)
    try:
      x = jnp.broadcast_to(x, target_shape)
    except Exception as e:
      # If broadcasting fails, try to handle the case where x has an extra dimension
      # that needs to be reduced (e.g., (200, 50, 10, 128) -> (200, 50, 128))
      if x.ndim == ref.ndim + 1:
        # Try squeezing any unit dimension that's not the last one
        for ax in range(x.ndim - 1):
          if x.shape[ax] == 1:
            x_squeezed = jnp.squeeze(x, axis=ax)
            try:
              x = jnp.broadcast_to(x_squeezed, target_shape)
              return x
            except Exception:
              # If this axis didn't work, continue to next axis
              continue
        
        # If no unit dimensions found, try taking the mean along the extra dimension
        # This handles cases like (200, 50, 10, 128) -> (200, 50, 128)
        if x.ndim == ref.ndim + 1:
          # Find the extra dimension (the one that's not in the reference)
          # We'll take the mean along the second-to-last dimension
          x_reduced = jnp.mean(x, axis=-2)
          try:
            x = jnp.broadcast_to(x_reduced, target_shape)
            return x
          except Exception:
            pass
      
      # If we get here, none of the reduction attempts worked
      raise ValueError(
        f"{name}: cannot broadcast from {x.shape} to {target_shape} "
        f"(ref={ref.shape})"
      ) from e
    return x

  def __call__(self, graph: jraph.GraphsTuple) -> jraph.ArrayTree:
    """Forward pass of the GNN."""

    # ---------- update fns ----------
    def update_global_fn(
        aggregated_nodes,
        aggregated_edges,
        globals_
    ):
        """
        Global updates operate on pooled per-graph features. Align ranks to the
        reference (aggregated_nodes), then concat on the feature axis.
        """
        ref = aggregated_nodes
        nodes_b = self._broadcast_to_ref(aggregated_nodes, ref, "aggregated_nodes")
        edges_b = self._broadcast_to_ref(aggregated_edges, ref, "aggregated_edges")
        globs_b = self._broadcast_to_ref(globals_,         ref, "globals")

        concatenated_features = jnp.concatenate([nodes_b, edges_b, globs_b], axis=-1)
        transformed_global = hk.Linear(output_size=self.embedding_size, name='global_fn_linear')(concatenated_features)
        transformed_global = jax.nn.relu(transformed_global)
        # residual on aligned globals
        chex.assert_equal_shape([globs_b, transformed_global])
        return globs_b + transformed_global
    
    def update_edge_fn(
        edge_features,
        sender_features,
        receiver_features,
        globals_
    ) -> Array:
        """
        Edge update must stay **per-edge**. Use `edge_features` as reference so the
        output keeps the same leading shape as edges. Broadcast the others to match.
        """
        ref = edge_features  # <-- per-edge reference
        edge_b     = self._broadcast_to_ref(edge_features,     ref, "edge_features")
        sender_b   = self._broadcast_to_ref(sender_features,   ref, "sender_features")
        receiver_b = self._broadcast_to_ref(receiver_features, ref, "receiver_features")

        parts = [edge_b, sender_b, receiver_b]
        if self.embed_globals:
            globals_b = self._broadcast_to_ref(globals_, ref, "globals")
            parts.append(globals_b)

        concatenated_features = jnp.concatenate(parts, axis=-1)
        transformed_edge = hk.Linear(output_size=self.embedding_size, name='edge_fn_linear')(concatenated_features)
        ln = hk.LayerNorm(axis=0, create_scale=True, create_offset=True, name='edge_fn_ln')

        if self.normalization == 'pre':
            transformed_edge = ln(transformed_edge)
            transformed_edge = jax.nn.relu(transformed_edge)
        elif self.normalization == 'post':
            transformed_edge = jax.nn.relu(transformed_edge)
            transformed_edge = ln(transformed_edge)
        elif self.normalization == 'none':
            transformed_edge = jax.nn.relu(transformed_edge)
        else:
            raise ValueError(f"Unknown normalization {self.normalization}")

        # Residual: keep per-edge shape
        chex.assert_equal_shape([edge_b, transformed_edge])
        return edge_b + transformed_edge

    def update_node_fn(
        node_features, 
        sent_attributes, 
        received_attributes, 
        globals_
    ) -> Array:
        """
        Node update must stay **per-node**. Use `node_features` as reference and
        broadcast sent/received/global attributes to match it.
        """
        ref = node_features  # <-- per-node reference
        node_b = self._broadcast_to_ref(node_features,      ref, "node_features")
        sent_b = self._broadcast_to_ref(sent_attributes,    ref, "sent_attributes")
        recv_b = self._broadcast_to_ref(received_attributes,ref, "received_attributes")

        parts = [node_b, sent_b, recv_b]
        if self.embed_globals:
            globals_b = self._broadcast_to_ref(globals_, ref, "globals")
            parts.append(globals_b)

        concatenated_features = jnp.concatenate(parts, axis=-1)
        transformed_node = hk.Linear(output_size=self.embedding_size, name='node_fn_linear')(concatenated_features)
        ln = hk.LayerNorm(axis=0, create_scale=True, create_offset=True, name='node_fn_ln')

        if self.normalization == 'pre':
            transformed_node = ln(transformed_node)
            transformed_node = jax.nn.relu(transformed_node)
        elif self.normalization == 'post':
            transformed_node = jax.nn.relu(transformed_node)
            transformed_node = ln(transformed_node)
        elif self.normalization == 'none':
            transformed_node = jax.nn.relu(transformed_node)
        else:
            raise ValueError(f"Unknown normalization {self.normalization}")

        # Residual: keep per-node shape
        chex.assert_equal_shape([node_b, transformed_node])
        return node_b + transformed_node

    # ---------- embeddings ----------
    embedding = jraph.GraphMapFeatures(
        embed_edge_fn=hk.Linear(output_size=self.embedding_size, name='edge_embedding'),
        embed_node_fn=hk.Linear(output_size=self.embedding_size, name='node_embedding'),
        embed_global_fn=hk.Linear(output_size=self.embedding_size, name='global_embedding') if self.embed_globals else None,
    )
    
    graph = embedding(graph)

    # ---------- message-passing blocks ----------
    graph = hk.Sequential([
        GraphNetwork(
            update_edge_fn=update_edge_fn,
            update_node_fn=update_node_fn,
            update_global_fn=update_global_fn if self.update_globals and self.embed_globals else None,
            aggregate_edges_for_globals_fn=self.aggregate_fn,
            aggregate_nodes_for_globals_fn=self.aggregate_fn,
            aggregate_edges_for_nodes_fn=self.aggregate_fn,
        )
        for _ in range(self.num_layers)
    ], name='GNN_Blocks')(graph)

    # ---------- decoders ----------
    if self.decode_globals:
        decoded_globals = hk.Linear(output_size=self.decode_global_dimension, name="global_decoder")(graph.globals)
        graph = graph._replace(globals=decoded_globals)
    
    if self.decode_edges:
        decoded_edges = hk.Linear(output_size=self.decode_edge_dimension, name="edge_decoder")(graph.edges)
        graph = graph._replace(edges=decoded_edges)

    return graph
