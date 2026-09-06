import numpy as np

def kv_cache_attention_step(x_new: np.ndarray, W_Q: np.ndarray, W_K: np.ndarray, W_V: np.ndarray, cache: tuple) -> tuple:
    """
    Perform a single attention step with KV caching.
    
    Args:
        x_new: New token embedding, shape (d_model,)
        W_Q: Query projection matrix, shape (d_model, d_k)
        W_K: Key projection matrix, shape (d_model, d_k)
        W_V: Value projection matrix, shape (d_model, d_v)
        cache: Tuple (K_cache, V_cache) or None if first step
    
    Returns:
        Tuple (output, updated_cache)
    """
    _, d_k = W_K.shape
    q_new = x_new @ W_Q
    k_new = x_new @ W_K
    v_new = x_new @ W_V

    if cache is None:
        k_cache = np.array([k_new])
        v_cache = np.array([v_new])
    else:
        k_cache, v_cache = cache

        k_cache = np.vstack([k_cache, k_new])
        v_cache = np.vstack([v_cache, v_new])

    similarity_score = np.divide( k_cache @ q_new , np.sqrt( d_k ) )
    softmax_score = np.divide( np.exp( similarity_score ), np.sum( np.exp( similarity_score ) ) )

    attention_score = softmax_score @ v_cache
    updated_cache = (k_cache, v_cache)

    return attention_score, updated_cache