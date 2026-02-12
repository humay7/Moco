import numpy as np
import jax
import jraph
import jax.numpy as jnp

class NumpyReplayBuffer:
    def __init__(self, capacity, dummy_batch):
        """
        Args:
            capacity: Max number of transitions to store.
            dummy_batch: A sample batch (pytree) from which to infer structure and shapes.
                         Should contain a batch dimension.
        """
        self.capacity = capacity
        self.size = 0
        self.ptr = 0
        
        # Flatten the dummy batch to get the leaves
        self.structure = jax.tree_util.tree_structure(dummy_batch)
        leaves, _ = jax.tree_util.tree_flatten(dummy_batch)
        
        # Create a buffer for each leaf
        # We assume the first dimension of each leaf is the batch dimension
        self.buffers = []
        for leaf in leaves:
            # leaf shape: (B, ...) -> storage shape: (capacity, ...)
            shape = (capacity,) + leaf.shape[1:]
            dtype = leaf.dtype
            self.buffers.append(np.zeros(shape, dtype=dtype))
            
    def add(self, batch):
        """
        Add a batch of transitions to the buffer.
        """
        leaves, _ = jax.tree_util.tree_flatten(batch)
        batch_size = leaves[0].shape[0]
        
        # If batch is larger than capacity, just take the last capacity elements
        if batch_size > self.capacity:
            batch_size = self.capacity
            leaves = [l[-self.capacity:] for l in leaves]
        
        # Handle wrap-around
        indices = np.arange(self.ptr, self.ptr + batch_size) % self.capacity
        
        for buf, leaf in zip(self.buffers, leaves):
            # Verify batch size consistency for this leaf
            if leaf.shape[0] != batch_size:
                raise ValueError(f"Batch dimension mismatch: expected {batch_size}, got {leaf.shape[0]} for leaf with shape {leaf.shape}")
            buf[indices] = leaf
            
        self.ptr = (self.ptr + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)
        
    def sample(self, batch_size):
        """
        Sample a batch of transitions.
        """
        if self.size == 0:
            raise ValueError("Buffer is empty")
            
        indices = np.random.randint(0, self.size, size=batch_size)
        
        result_leaves = [buf[indices] for buf in self.buffers]
        
        return jax.tree_util.tree_unflatten(self.structure, result_leaves)

    def __len__(self):
        return self.size

class GraphReplayBuffer:
    def __init__(self, capacity, dummy_batch_flat, num_nodes, num_edges):
        """
        Wrapper around NumpyReplayBuffer that handles reshaping of GraphsTuple 
        components to/from (Batch, N, ...) format for storage.
        
        Args:
            capacity: Buffer size.
            dummy_batch_flat: A sample flattened batch (from collect_episodes_batched).
            num_nodes: Number of nodes per graph.
            num_edges: Number of edges per graph.
        """
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        
        # Reshape dummy batch to be "unflattened" (item-based) for buffer initialization
        dummy_batch_unflat = self._unflatten_batch(dummy_batch_flat)
        
        self.buffer = NumpyReplayBuffer(capacity, dummy_batch_unflat)

    def _unflatten_batch(self, batch):
        """
        Convert flattened batch (where nodes are B*N) to unflattened batch (where nodes are B, N).
        Assumes batch is a tuple/list matching the return of collect_episodes_batched:
        (states, actions, rewards, next_states, dones, timesteps, stats)
        """
        states, actions, rewards, next_states, dones, timesteps, stats = batch
        
        # Helper to unflatten a GraphsTuple
        def unflat_graph(g):
            # nodes: (B*N, F) -> (B, N, F)
            B_times_N = g.nodes.shape[0]
            B = B_times_N // self.num_nodes
            
            nodes = g.nodes.reshape(B, self.num_nodes, -1)
            edges = g.edges.reshape(B, self.num_edges, -1)
            globals_ = g.globals # Should be (B, F) already if came from collect_episodes_batched
            
            # Reconstruct senders/receivers not needed for storage if we assume fixed topology
            # But we store them to be safe or just ignore? 
            # The flattened senders are (B*E,). We can reshape to (B, E).
            senders = g.senders.reshape(B, self.num_edges)
            receivers = g.receivers.reshape(B, self.num_edges)
            
            n_node = g.n_node.reshape(B)
            n_edge = g.n_edge.reshape(B)
            
            return jraph.GraphsTuple(
                nodes=nodes, edges=edges, globals=globals_,
                senders=senders, receivers=receivers,
                n_node=n_node, n_edge=n_edge
            )

        states_unflat = unflat_graph(states)
        next_states_unflat = unflat_graph(next_states)
        
        # Actions: (B*E, 1) -> (B, E, 1)
        actions_unflat = actions.reshape(-1, self.num_edges, 1)
        
        # Rewards, Dones, Timesteps: (B, 1) or (B,) -> Keep as is
        # Stats: Dict of flattened arrays. Reshape them too?
        # stats is flattened (E*T,). Keep as is.
        
        return (states_unflat, actions_unflat, rewards, next_states_unflat, dones, timesteps, stats)

    def _flatten_batch(self, batch):
        """
        Convert unflattened batch (from buffer) back to flattened batch (for jraph/network).
        """
        states_u, actions_u, rewards, next_states_u, dones, timesteps, stats = batch
        
        B = rewards.shape[0]
        
        def flat_graph(g):
            nodes = g.nodes.reshape(B * self.num_nodes, -1)
            edges = g.edges.reshape(B * self.num_edges, -1)
            globals_ = g.globals
            
            # Recalculate senders/receivers with offsets
            # stored senders was (B, E) relative to 0..N-1?
            # actually unflatten_batch just reshaped senders.
            # but wait, collect_episodes_batched produces global offsets.
            # "senders = (senders_ETe.reshape(-1) + node_offsets)"
            # So the stored senders in buffer are globally offset for the collected batch!
            # WARN: If we mix batches, the offsets will be wrong.
            # We must normalize senders to 0..N-1 before storage, and re-offset after sampling.
            
            # Let's assume we fix senders/receivers upon sampling.
            # We can ignore stored senders/receivers and regenerate them.
            
            senders_local = jnp.tile(jnp.arange(self.num_nodes), self.num_edges // self.num_nodes) # APPROX? No.
            # Better: The topology is fixed (TSP).
            # But simpler: collect_episodes_batched constructs senders/receivers from scratch anyway.
            # We can replicate that logic.
            
            # Stored senders in unflat_graph were just reshaped.
            # We need to correctly offset them for the NEW batch size.
            
            # Actually, let's just use the logic from collect_episodes_batched to make fresh senders/receivers
            # assuming fully connected or k-NN.
            # Wait, k-NN graph topology might vary per instance?
            # TSPTaskFamily: "graph = knn_graph(problem, k)". 
            # So topology DEPENDS on the instance (problem coordinates).
            # We MUST store the topology (senders/receivers).
            
            # Storage: (B, E). Values are global offsets?
            # When we flattened them in _unflatten_batch, we took global offsets.
            # e.g. graph 0: 0..N. graph 1: N..2N.
            # If we store these, valid for that batch.
            # When we sample, we might mix graph 0 from batch A and graph 5 from batch B.
            # Their offsets will be wrong relative to the new sampled batch 0..B.
            
            # FIX: Convert global offsets to local offsets (0..N-1) before storage.
            # Local = Global % num_nodes.
            
            senders_local = g.senders % self.num_nodes
            receivers_local = g.receivers % self.num_nodes
            
            # Now add new offsets for the sampled batch
            offsets = jnp.arange(B) * self.num_nodes
            offsets = jnp.repeat(offsets, self.num_edges) # (B*E,)
            
            senders_flat = (senders_local.flatten() + offsets).astype(jnp.int32)
            receivers_flat = (receivers_local.flatten() + offsets).astype(jnp.int32)
            
            n_node = jnp.full((B,), self.num_nodes)
            n_edge = jnp.full((B,), self.num_edges)
            
            return jraph.GraphsTuple(
                nodes=nodes, edges=edges, globals=globals_,
                senders=senders_flat, receivers=receivers_flat,
                n_node=n_node, n_edge=n_edge
            )

        states_f = flat_graph(states_u)
        next_states_f = flat_graph(next_states_u)
        
        actions_f = actions_u.reshape(B * self.num_edges, -1)
        
        return (states_f, actions_f, rewards, next_states_f, dones, timesteps, stats)

    def add(self, batch):
        batch_unflat = self._unflatten_batch(batch)
        self.buffer.add(batch_unflat)

    def sample(self, batch_size):
        batch_unflat = self.buffer.sample(batch_size)
        return self._flatten_batch(batch_unflat)

    def __len__(self):
        return len(self.buffer)
