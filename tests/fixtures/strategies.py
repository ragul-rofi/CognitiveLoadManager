"""Hypothesis strategies for generating test data."""

from datetime import datetime, timedelta
from hypothesis import strategies as st
from clm.core.models import TaskNode, TaskTree, Signals, TaskChunk


@st.composite
def task_node_strategy(draw, parent_id=None, depth=0, max_depth=3):
    """
    Generate a TaskNode with random but valid data.
    
    Args:
        parent_id: Parent task ID (None for root)
        depth: Current depth in tree
        max_depth: Maximum depth to generate children
    
    Returns:
        TaskNode instance
    """
    task_id = draw(st.text(min_size=1, max_size=20, alphabet=st.characters(
        whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='-_'
    )))
    description = draw(st.text(min_size=10, max_size=500))
    status = draw(st.sampled_from(["active", "compressed", "completed"]))
    
    node = TaskNode(
        task_id=task_id,
        parent_id=parent_id,
        description=description,
        status=status,
        depth=depth,
        children=[]
    )
    
    # Recursively generate children if not at max depth
    if depth < max_depth:
        num_children = draw(st.integers(min_value=0, max_value=3))
        for _ in range(num_children):
            child = draw(task_node_strategy(
                parent_id=task_id,
                depth=depth + 1,
                max_depth=max_depth
            ))
            node.children.append(child)
    
    return node


@st.composite
def task_tree_strategy(draw, max_depth=3, min_nodes=1, max_nodes=10):
    """
    Generate a TaskTree with configurable depth and branching.
    
    Args:
        max_depth: Maximum depth of the tree
        min_nodes: Minimum number of nodes in tree
        max_nodes: Maximum number of nodes in tree
    
    Returns:
        TaskTree instance
    """
    root = draw(task_node_strategy(parent_id=None, depth=0, max_depth=max_depth))
    root_intent = draw(st.text(min_size=10, max_size=200))
    
    # Optionally generate root intent embedding (384-dimensional for all-MiniLM-L6-v2)
    has_embedding = draw(st.booleans())
    root_intent_embedding = None
    if has_embedding:
        root_intent_embedding = draw(st.lists(
            st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=384,
            max_size=384
        ))
    
    return TaskTree(
        root=root,
        root_intent=root_intent,
        root_intent_embedding=root_intent_embedding
    )


@st.composite
def signals_strategy(draw):
    """
    Generate Signals with all values in [0, 1] range.
    
    Returns:
        Signals instance with normalized values
    """
    branching_factor = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    repetition_rate = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    uncertainty_density = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    goal_distance = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    
    return Signals(
        branching_factor=branching_factor,
        repetition_rate=repetition_rate,
        uncertainty_density=uncertainty_density,
        goal_distance=goal_distance
    )


@st.composite
def weights_strategy(draw, num_weights=4):
    """
    Generate weight arrays that sum to 1.0.
    
    Args:
        num_weights: Number of weights to generate (default 4 for CLM)
    
    Returns:
        List of floats that sum to 1.0
    """
    # Generate random positive values
    raw_weights = [draw(st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False)) 
                   for _ in range(num_weights)]
    
    # Normalize to sum to 1.0
    total = sum(raw_weights)
    normalized_weights = [w / total for w in raw_weights]
    
    return normalized_weights


@st.composite
def llm_output_strategy(draw, hedged_token_density=None, min_tokens=50, max_tokens=500):
    """
    Generate LLM output text with configurable hedged token density.
    
    Args:
        hedged_token_density: Target density of hedged tokens (0-1), or None for random
        min_tokens: Minimum number of tokens in output
        max_tokens: Maximum number of tokens in output
    
    Returns:
        String representing LLM output with specified hedged token density
    """
    hedged_tokens = [
        "maybe", "perhaps", "possibly", "might", "could",
        "uncertain", "unclear", "probably", "likely", "seems"
    ]
    
    # Regular words for filler
    regular_words = [
        "the", "task", "requires", "implementation", "system", "function",
        "process", "data", "result", "value", "method", "class", "object",
        "user", "input", "output", "error", "success", "complete", "start",
        "end", "begin", "finish", "create", "update", "delete", "read", "write"
    ]
    
    # Determine target density
    if hedged_token_density is None:
        target_density = draw(st.floats(min_value=0.0, max_value=0.3, allow_nan=False, allow_infinity=False))
    else:
        target_density = hedged_token_density
    
    # Generate token count
    num_tokens = draw(st.integers(min_value=min_tokens, max_value=max_tokens))
    
    # Calculate number of hedged tokens to include
    num_hedged = int(num_tokens * target_density)
    num_regular = num_tokens - num_hedged
    
    # Generate tokens
    tokens = []
    
    # Add hedged tokens
    for _ in range(num_hedged):
        tokens.append(draw(st.sampled_from(hedged_tokens)))
    
    # Add regular tokens
    for _ in range(num_regular):
        tokens.append(draw(st.sampled_from(regular_words)))
    
    # Shuffle to mix hedged and regular tokens
    draw(st.randoms()).shuffle(tokens)
    
    # Join into sentences (roughly 10-15 words per sentence)
    output_parts = []
    current_sentence = []
    
    for i, token in enumerate(tokens):
        current_sentence.append(token)
        
        # End sentence every 10-15 words
        if len(current_sentence) >= 10 and (i == len(tokens) - 1 or draw(st.booleans())):
            sentence = " ".join(current_sentence).capitalize() + "."
            output_parts.append(sentence)
            current_sentence = []
    
    # Add any remaining tokens
    if current_sentence:
        sentence = " ".join(current_sentence).capitalize() + "."
        output_parts.append(sentence)
    
    return " ".join(output_parts)


@st.composite
def task_chunk_strategy(draw):
    """
    Generate a TaskChunk for sidecar storage testing.
    
    Returns:
        TaskChunk instance
    """
    task_id = draw(st.text(min_size=1, max_size=20, alphabet=st.characters(
        whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='-_'
    )))
    
    # Parent ID can be None or a string
    has_parent = draw(st.booleans())
    parent_id = None
    if has_parent:
        parent_id = draw(st.text(min_size=1, max_size=20, alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='-_'
        )))
    
    summary = draw(st.text(min_size=10, max_size=200))
    full_detail = draw(st.text(min_size=50, max_size=1000))
    clm_score = draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False))
    
    # Generate timestamp within last 30 days
    days_ago = draw(st.integers(min_value=0, max_value=30))
    compressed_at = datetime.now() - timedelta(days=days_ago)
    
    status = draw(st.sampled_from(["compressed", "expanded"]))
    
    return TaskChunk(
        task_id=task_id,
        parent_id=parent_id,
        summary=summary,
        full_detail=full_detail,
        clm_score_at_compression=clm_score,
        compressed_at=compressed_at,
        status=status
    )
