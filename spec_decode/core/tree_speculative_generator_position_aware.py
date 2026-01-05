"""
Position-Aware Tree Speculative Generator.

This module provides a modified version of TreeSpeculativeGeneratorV2 that
correctly handles position encoding for RoPE-based models during tree
verification.

Problem:
    When the tree is flattened for parallel verification, all nodes get
    sequential position IDs (prefix_len, prefix_len+1, prefix_len+2, ...).
    However, nodes at the same depth should share the same position ID
    because they represent the same semantic position in the sequence.
    
    For RoPE-based models (like Pythia, LLaMA), incorrect position IDs
    lead to wrong rotary embeddings and thus incorrect predictions.

Solution:
    This generator passes explicit position_ids to the target model during
    verification, where nodes at the same depth share the same position ID
    (prefix_len + depth).
"""

import torch
from typing import Tuple

from .tree_speculative_generator import TreeSpeculativeGeneratorV2
from .token_tree import TokenTree


class TreeSpeculativeGeneratorV2PositionAware(TreeSpeculativeGeneratorV2):
    """
    Tree-based Speculative Decoding Generator with Position-Aware Verification.
    
    This version correctly handles RoPE position encoding during tree
    verification by passing explicit position_ids to the target model.
    
    Key difference from TreeSpeculativeGeneratorV2:
        - Uses flatten_for_verification_with_positions() to get depth-based positions
        - Passes position_ids to target model during verification
        - Nodes at the same tree depth share the same position ID
    
    This should improve acceptance rates for RoPE-based models, especially
    for larger trees where position displacement is more significant.
    """
    
    @torch.inference_mode()
    def _verify_tree_tokens(
        self,
        tree: TokenTree
    ) -> Tuple[torch.Tensor, torch.Tensor, object]:
        """
        Verify all tree tokens with the target model using position-aware tree attention.
        
        This method differs from the parent by:
        1. Getting depth-based position IDs from the tree
        2. Adding prefix_len to get actual position IDs
        3. Passing position_ids explicitly to the target model
        
        Args:
            tree: TokenTree containing draft candidates
            
        Returns:
            target_logits: Logits from target model [num_nodes, vocab_size]
            attention_mask: Tree attention mask used
            verify_outputs: Original model outputs
        """
        # Flatten tree for verification WITH position IDs
        tree_tokens, node_indices, tree_depths = tree.flatten_for_verification_with_positions()
        
        # Get current cache length (prefix length)
        prefix_len = self.target_cache.get_seq_length()
        
        # Calculate actual position IDs: prefix_len + depth
        # This ensures nodes at the same depth share the same position ID
        position_ids = tree_depths + prefix_len  # [num_nodes]
        
        # Build tree attention mask
        # Shape: [num_nodes, prefix_len + num_nodes]
        tree_mask = tree.build_tree_attention_mask(prefix_len=prefix_len)
        
        # Convert to the format expected by HuggingFace models
        num_nodes = len(tree_tokens)
        total_len = prefix_len + num_nodes
        
        # Create 4D attention mask [1, 1, num_nodes, total_len]
        # Convert bool mask to float: True -> 0.0, False -> -inf
        attention_mask_4d = torch.zeros(
            (1, 1, num_nodes, total_len),
            dtype=self.target_model.dtype if hasattr(self.target_model, 'dtype') else torch.float16,
            device=self.device
        )
        attention_mask_4d[0, 0] = torch.where(
            tree_mask,
            torch.tensor(0.0, device=self.device),
            torch.tensor(float('-inf'), device=self.device)
        )
        
        # Forward tree tokens through target model with explicit position_ids
        tree_tokens_input = tree_tokens.unsqueeze(0)  # [1, num_nodes]
        position_ids_input = position_ids.unsqueeze(0)  # [1, num_nodes]
        
        try:
            outputs = self.target_model(
                input_ids=tree_tokens_input,
                position_ids=position_ids_input,  # Key difference: explicit position IDs
                past_key_values=self.target_cache,
                attention_mask=attention_mask_4d,
                use_cache=True,
                return_dict=True
            )
        except Exception as e:
            # Fallback: some models don't support 4D attention mask or position_ids
            # Use sequential verification instead (which has correct positions naturally)
            return self._verify_tree_sequential(tree)
        
        # Extract logits for each position
        # outputs.logits shape: [1, num_nodes, vocab_size]
        target_logits = outputs.logits[0]  # [num_nodes, vocab_size]
        
        # Store the new cache (will be adjusted later based on accepted path)
        self._verify_cache = outputs.past_key_values
        
        return target_logits, tree_mask, outputs


__all__ = ["TreeSpeculativeGeneratorV2PositionAware"]


