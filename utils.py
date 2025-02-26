# utils.py
import torch

def ensure_features(data):
    """
    Ensures a PyTorch Geometric Data object has valid node features (x).
    If x is None, creates a dummy feature of shape [num_nodes, 1].
    """
    if data.x is None:
        num_nodes = data.num_nodes
        data.x = torch.ones((num_nodes, 1), dtype=torch.float)
    return data

def state_to_tensor(state):
    """
    Flattens node features into a 1D tensor for value-based methods.
    """
    return state.x.view(-1)

def _safe_tensor_edges(edges_list):
    """
    Converts an edge list (a list of [source, target] pairs) into a torch tensor of shape [2, E],
    or returns an empty tensor if there are no edges.
    """
    if len(edges_list) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    e_t = torch.tensor(edges_list, dtype=torch.long)
    if e_t.dim() == 2 and e_t.size(1) == 2:
        return e_t.t().contiguous()
    else:
        return torch.empty((2, 0), dtype=torch.long)
