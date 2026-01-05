"""
Tree Speculative Generator with Sequential Verification.

This is a modified version that uses sequential verification instead of
tree attention to avoid RoPE position encoding issues.
"""

from .tree_speculative_generator import TreeSpeculativeGeneratorV2


class TreeSpeculativeGeneratorV2Sequential(TreeSpeculativeGeneratorV2):
    """
    Tree-based Speculative Decoding Generator with Sequential Verification.
    
    This version overrides _verify_tree_tokens to always use sequential
    verification, which correctly handles position encoding for RoPE-based
    models like Pythia and LLaMA.
    
    The issue with tree attention: when the tree is flattened, the
    position_in_sequence doesn't reflect the actual semantic position.
    This causes RoPE to give incorrect position embeddings, leading to
    wrong predictions especially for larger trees (more nodes = more
    position displacement).
    """
    
    def _verify_tree_tokens(self, tree):
        """
        Verify all tree tokens using sequential verification.
        
        Instead of verifying all nodes at once with tree attention mask,
        this method verifies each leaf path independently. This ensures
        correct position encoding for each path.
        
        Args:
            tree: TokenTree containing draft candidates
            
        Returns:
            target_logits: Logits from target model [num_nodes, vocab_size]
            attention_mask: Tree attention mask (for compatibility)
            verify_outputs: None (not used in sequential mode)
        """
        return self._verify_tree_sequential(tree)



