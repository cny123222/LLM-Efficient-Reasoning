"""
Token Tree Data Structure for Tree-based Speculative Decoding

This module provides data structures for managing token trees in tree-based
speculative decoding (SpecInfer-style). The tree structure allows the draft
model to generate multiple candidate sequences, which are then verified in
parallel by the target model.

Key Components:
- TreeNode: Individual node in the token tree
- TokenTree: Complete tree structure with traversal and manipulation methods
"""

import torch
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, field


@dataclass
class TreeNode:
    """
    A node in the token tree.
    
    Each node represents a token at a specific position in a candidate sequence.
    Nodes maintain parent-child relationships to form the tree structure.
    
    Attributes:
        token_id: The token ID for this node
        parent_idx: Index of parent node in the tree's node list (-1 for root)
        children_idx: Indices of child nodes in the tree's node list
        depth: Depth in the tree (0 for root)
        logit: Log probability of this token from the draft model
        cumulative_logit: Sum of logits from root to this node
        position_in_sequence: Position of this token in the flattened sequence
    """
    token_id: int
    parent_idx: int = -1
    children_idx: List[int] = field(default_factory=list)
    depth: int = 0
    logit: float = 0.0
    cumulative_logit: float = 0.0
    position_in_sequence: int = -1  # Position in flattened sequence for verification
    
    def is_root(self) -> bool:
        """Check if this is the root node."""
        return self.parent_idx == -1
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node (no children)."""
        return len(self.children_idx) == 0
    
    def add_child(self, child_idx: int):
        """Add a child node index."""
        self.children_idx.append(child_idx)


class TokenTree:
    """
    Token tree for tree-based speculative decoding.
    
    The tree is built during the draft phase, where the draft model generates
    multiple candidate tokens at each position (using top-k sampling). The tree
    is then flattened into a sequence for parallel verification by the target model.
    
    Tree Structure Example:
    ```
              root (prompt's last token prediction)
             / | \\
            t1 t2 t3     (depth 1, branch_factor candidates)
           /|  |  |\\
         t4 t5 t6 t7 t8  (depth 2)
    ```
    
    The tree is stored as a list of TreeNode objects, with parent-child relationships
    tracked via indices.
    
    Args:
        max_depth: Maximum depth of the tree (equivalent to K in linear decoding)
        branch_factor: Number of branches per node (top-k)
        max_nodes: Maximum number of nodes in the tree
        device: Device for tensor operations
    """
    
    def __init__(
        self,
        max_depth: int = 4,
        branch_factor: int = 2,
        max_nodes: int = 32,
        device: str = "cuda"
    ):
        self.max_depth = max_depth
        self.branch_factor = branch_factor
        self.max_nodes = max_nodes
        self.device = device
        
        # Node storage
        self.nodes: List[TreeNode] = []
        
        # Mapping from (parent_idx, depth) to node indices for fast lookup
        self._depth_nodes: Dict[int, List[int]] = {}
        
        # Flattened sequence for verification (computed lazily)
        self._flattened_tokens: Optional[torch.Tensor] = None
        self._flattened_positions: Optional[List[int]] = None
        self._attention_mask: Optional[torch.Tensor] = None
        
    def reset(self):
        """Reset the tree for a new draft round."""
        self.nodes = []
        self._depth_nodes = {}
        self._flattened_tokens = None
        self._flattened_positions = None
        self._attention_mask = None
    
    def add_root(self, token_id: int, logit: float = 0.0) -> int:
        """
        Add the root node (represents the prediction for the first draft token).
        
        Args:
            token_id: Token ID for root
            logit: Log probability of this token
            
        Returns:
            Index of the root node (always 0)
        """
        assert len(self.nodes) == 0, "Root already exists"
        
        root = TreeNode(
            token_id=token_id,
            parent_idx=-1,
            depth=0,
            logit=logit,
            cumulative_logit=logit
        )
        self.nodes.append(root)
        self._depth_nodes[0] = [0]
        self._invalidate_cache()
        return 0
    
    def add_node(
        self,
        token_id: int,
        parent_idx: int,
        logit: float = 0.0
    ) -> int:
        """
        Add a new node to the tree.
        
        Args:
            token_id: Token ID for the new node
            parent_idx: Index of the parent node
            logit: Log probability of this token
            
        Returns:
            Index of the new node
        """
        assert len(self.nodes) < self.max_nodes, f"Tree is full ({self.max_nodes} nodes)"
        assert 0 <= parent_idx < len(self.nodes), f"Invalid parent index: {parent_idx}"
        
        parent = self.nodes[parent_idx]
        new_depth = parent.depth + 1
        
        if new_depth > self.max_depth:
            raise ValueError(f"Cannot add node at depth {new_depth}, max is {self.max_depth}")
        
        node_idx = len(self.nodes)
        node = TreeNode(
            token_id=token_id,
            parent_idx=parent_idx,
            depth=new_depth,
            logit=logit,
            cumulative_logit=parent.cumulative_logit + logit
        )
        
        self.nodes.append(node)
        parent.add_child(node_idx)
        
        # Update depth index
        if new_depth not in self._depth_nodes:
            self._depth_nodes[new_depth] = []
        self._depth_nodes[new_depth].append(node_idx)
        
        self._invalidate_cache()
        return node_idx
    
    def add_children(
        self,
        parent_idx: int,
        token_ids: torch.Tensor,
        logits: torch.Tensor
    ) -> List[int]:
        """
        Add multiple children to a parent node.
        
        Args:
            parent_idx: Index of the parent node
            token_ids: Token IDs for children [num_children]
            logits: Log probabilities for children [num_children]
            
        Returns:
            List of indices for the new children nodes
        """
        child_indices = []
        for i in range(len(token_ids)):
            token_id = token_ids[i].item()
            logit = logits[i].item() if logits is not None else 0.0
            idx = self.add_node(token_id, parent_idx, logit)
            child_indices.append(idx)
        return child_indices
    
    def get_nodes_at_depth(self, depth: int) -> List[int]:
        """Get indices of all nodes at a specific depth."""
        return self._depth_nodes.get(depth, [])
    
    def get_path_to_root(self, node_idx: int) -> List[int]:
        """
        Get the path from a node to the root (inclusive).
        
        Args:
            node_idx: Index of the target node
            
        Returns:
            List of node indices from root to the target node
        """
        path = []
        current_idx = node_idx
        
        while current_idx != -1:
            path.append(current_idx)
            current_idx = self.nodes[current_idx].parent_idx
        
        return list(reversed(path))
    
    def get_path_tokens(self, node_idx: int) -> List[int]:
        """
        Get token IDs along the path from root to a node.
        
        Args:
            node_idx: Index of the target node
            
        Returns:
            List of token IDs from root to the target node
        """
        path = self.get_path_to_root(node_idx)
        return [self.nodes[idx].token_id for idx in path]
    
    def get_leaf_nodes(self) -> List[int]:
        """Get indices of all leaf nodes."""
        return [i for i, node in enumerate(self.nodes) if node.is_leaf()]
    
    def get_all_paths(self) -> List[List[int]]:
        """
        Get all paths from root to leaf nodes.
        
        Returns:
            List of paths, where each path is a list of token IDs
        """
        leaves = self.get_leaf_nodes()
        return [self.get_path_tokens(leaf_idx) for leaf_idx in leaves]
    
    def _invalidate_cache(self):
        """Invalidate cached computations."""
        self._flattened_tokens = None
        self._flattened_positions = None
        self._attention_mask = None
    
    def flatten_for_verification(self) -> Tuple[torch.Tensor, List[int]]:
        """
        Flatten the tree into a sequence for parallel verification.
        
        The flattening order is BFS (breadth-first), which ensures that
        parent tokens appear before their children in the sequence.
        
        Returns:
            flattened_tokens: Token IDs in BFS order [num_nodes]
            node_indices: Mapping from position to node index
        """
        if self._flattened_tokens is not None:
            return self._flattened_tokens, self._flattened_positions
        
        # BFS traversal
        flattened_tokens = []
        node_indices = []
        
        for depth in range(self.max_depth + 1):
            nodes_at_depth = self.get_nodes_at_depth(depth)
            for node_idx in nodes_at_depth:
                node = self.nodes[node_idx]
                node.position_in_sequence = len(flattened_tokens)
                flattened_tokens.append(node.token_id)
                node_indices.append(node_idx)
        
        self._flattened_tokens = torch.tensor(
            flattened_tokens, dtype=torch.long, device=self.device
        )
        self._flattened_positions = node_indices
        
        return self._flattened_tokens, self._flattened_positions
    
    def build_tree_attention_mask(self, prefix_len: int = 0) -> torch.Tensor:
        """
        Build the tree attention mask for parallel verification.
        
        The mask encodes the parent-child relationships in the tree.
        Each token can only attend to its ancestors (path to root) and
        the prefix tokens.
        
        For tree:
        ```
                root
               / | \\
              t1 t2 t3
             /|  |  |\\
            t4 t5 t6 t7 t8
        ```
        
        The mask (for tree nodes only, prefix handled separately):
        ```
             root  t1  t2  t3  t4  t5  t6  t7  t8
        root   1   0   0   0   0   0   0   0   0
        t1     1   1   0   0   0   0   0   0   0
        t2     1   0   1   0   0   0   0   0   0
        t3     1   0   0   1   0   0   0   0   0
        t4     1   1   0   0   1   0   0   0   0
        t5     1   1   0   0   0   1   0   0   0
        t6     1   0   1   0   0   0   1   0   0
        t7     1   0   0   1   0   0   0   1   0
        t8     1   0   0   1   0   0   0   0   1
        ```
        
        Args:
            prefix_len: Length of the prefix (prompt) that all tokens attend to
            
        Returns:
            attention_mask: Boolean mask [num_nodes, prefix_len + num_nodes]
                           True means can attend, False means masked
        """
        if self._attention_mask is not None and prefix_len == 0:
            return self._attention_mask
        
        # Ensure tree is flattened
        _, node_indices = self.flatten_for_verification()
        num_nodes = len(node_indices)
        
        # Build position-to-node mapping
        pos_to_node = {self.nodes[idx].position_in_sequence: idx for idx in node_indices}
        
        # Initialize mask: all nodes attend to all prefix tokens
        total_len = prefix_len + num_nodes
        mask = torch.zeros((num_nodes, total_len), dtype=torch.bool, device=self.device)
        
        # All tree nodes can attend to prefix
        if prefix_len > 0:
            mask[:, :prefix_len] = True
        
        # Build tree attention pattern
        for i, node_idx in enumerate(node_indices):
            # Get path from this node to root
            path = self.get_path_to_root(node_idx)
            
            # This node can attend to all ancestors (including itself)
            for ancestor_idx in path:
                ancestor_pos = self.nodes[ancestor_idx].position_in_sequence
                mask[i, prefix_len + ancestor_pos] = True
        
        if prefix_len == 0:
            self._attention_mask = mask
        
        return mask
    
    def get_parent_indices(self) -> torch.Tensor:
        """
        Get parent indices for each node in flattened order.
        
        This is useful for gathering parent logits during verification.
        
        Returns:
            parent_indices: Index of parent in flattened sequence for each node
                           -1 for root node [num_nodes]
        """
        _, node_indices = self.flatten_for_verification()
        parent_indices = []
        
        for node_idx in node_indices:
            node = self.nodes[node_idx]
            if node.parent_idx == -1:
                parent_indices.append(-1)
            else:
                parent_pos = self.nodes[node.parent_idx].position_in_sequence
                parent_indices.append(parent_pos)
        
        return torch.tensor(parent_indices, dtype=torch.long, device=self.device)
    
    def __len__(self) -> int:
        """Return number of nodes in the tree."""
        return len(self.nodes)
    
    def __repr__(self) -> str:
        depths = {d: len(nodes) for d, nodes in self._depth_nodes.items()}
        return f"TokenTree(nodes={len(self.nodes)}, depths={depths}, max_depth={self.max_depth})"
    
    def visualize(self, tokenizer=None) -> str:
        """
        Create a string visualization of the tree.
        
        Args:
            tokenizer: Optional tokenizer to decode tokens
            
        Returns:
            String representation of the tree structure
        """
        if len(self.nodes) == 0:
            return "Empty tree"
        
        lines = []
        
        def _visualize_node(node_idx: int, prefix: str = "", is_last: bool = True):
            node = self.nodes[node_idx]
            connector = "└── " if is_last else "├── "
            
            if tokenizer is not None:
                token_str = tokenizer.decode([node.token_id])
                token_repr = f"'{token_str}' ({node.token_id})"
            else:
                token_repr = f"token={node.token_id}"
            
            lines.append(f"{prefix}{connector}{token_repr} [depth={node.depth}, logit={node.logit:.2f}]")
            
            child_prefix = prefix + ("    " if is_last else "│   ")
            children = node.children_idx
            for i, child_idx in enumerate(children):
                _visualize_node(child_idx, child_prefix, i == len(children) - 1)
        
        _visualize_node(0, "", True)
        return "\n".join(lines)


def build_tree_from_topk(
    logits_sequence: List[torch.Tensor],
    k: int,
    max_depth: int,
    device: str = "cuda"
) -> TokenTree:
    """
    Build a token tree from a sequence of logits using top-k selection.
    
    This is a utility function for building trees during the draft phase.
    At each depth, we expand each leaf node with top-k candidates.
    
    Args:
        logits_sequence: List of logits tensors, one per depth
                        Each tensor has shape [num_parents, vocab_size]
        k: Number of top candidates to select at each position
        max_depth: Maximum tree depth
        device: Device for tensors
        
    Returns:
        TokenTree with the constructed tree structure
    """
    tree = TokenTree(max_depth=max_depth, branch_factor=k, device=device)
    
    if len(logits_sequence) == 0:
        return tree
    
    # First level: single parent (prompt's last token)
    first_logits = logits_sequence[0]  # [1, vocab] or [vocab]
    if first_logits.dim() == 1:
        first_logits = first_logits.unsqueeze(0)
    
    # Get top-k for root
    log_probs = torch.log_softmax(first_logits[0], dim=-1)
    topk_logits, topk_tokens = torch.topk(log_probs, k)
    
    # Add root (first top-k candidate)
    tree.add_root(topk_tokens[0].item(), topk_logits[0].item())
    
    # Add siblings as children of an implicit root
    # Actually, for tree-based drafting, we typically have multiple roots
    # Let me restructure to handle this case properly
    
    # For simplicity, we use the most likely token as the single root
    # and expand from there. A more advanced implementation would
    # support multiple roots (forest).
    
    # Build subsequent levels
    for depth in range(1, min(len(logits_sequence), max_depth)):
        current_leaves = tree.get_nodes_at_depth(depth - 1)
        depth_logits = logits_sequence[depth]  # [num_leaves, vocab]
        
        if depth_logits.dim() == 1:
            depth_logits = depth_logits.unsqueeze(0)
        
        for i, leaf_idx in enumerate(current_leaves):
            if i >= depth_logits.shape[0]:
                break
            
            log_probs = torch.log_softmax(depth_logits[i], dim=-1)
            topk_logits, topk_tokens = torch.topk(log_probs, k)
            
            tree.add_children(leaf_idx, topk_tokens, topk_logits)
    
    return tree


__all__ = [
    "TreeNode",
    "TokenTree",
    "build_tree_from_topk"
]

