import torch
import torch.nn.functional as F


def similarity_aggregation(latents, instructions, mean_aggregation: bool = False, top_k_selection: bool = False, soft_selection_sigma: float = 0.1):
    """
    :param latents: [B, H*W*D, C]
    :param instructions: [B, I, N, C]
    :return: [B, I, H*W*D]
    """
    # Note: Segmenter (https://github.com/rstrudel/segmenter) effectively does a linear layer with weights separate for content and class tokens before doing the comparison (but without changing the dimension)
    # x_inst = self.linear_projection(dict_out['instructions']['segmentation_latents'])
    x_sim = torch.einsum('b m c, b i n c -> b i n m',
                         F.normalize(latents, p=2, dim=-1),
                         F.normalize(instructions, p=2, dim=-1)) + 1. / 2.  # Calculate similarities in range [0, 1] between instructions and content

    # (Post) selection of instructions
    assert mean_aggregation is False or top_k_selection is False  # Both can't be true at once
    if mean_aggregation:
        x_sim = torch.mean(x_sim, dim=2)
    elif top_k_selection:
        # Top k selection with k=3. 1. Doesn't have to align to all, 2. Single outlier (max) is prevented due to top k averaging.
        x_sim = torch.topk(x_sim, k=3, dim=2)[0]
        x_sim = torch.mean(x_sim, dim=2)  # Average similarities of top k tokens of the respective mask
    else:
        # Re-weight by relative importance (detached softmaxed similarities). 1. Doesn't have to align to all, 2. All instructions receive a (weighted) gradient.
        x_sim = torch.softmax(x_sim.detach() / soft_selection_sigma, dim=2) * x_sim
        x_sim = torch.sum(x_sim, dim=2)  # Aggregate weighted similarities

    return x_sim
