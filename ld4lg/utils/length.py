import torch


from collections import Counter
import torch

def get_length_distribution(dataset, max_seq_len):
    """
    Given a dataset, compute a length distribution for the sequence lengths found in the dataset.

    Args:
        dataset (datasets.arrow_dataset.Dataset): The dataset to analyze.
        max_seq_len (int): The maximum sequence length considered.

    Returns:
        length_distribution (torch.distributions.Categorical): A categorical distribution of sequence lengths.
    """

    lengths = [min(sum(dataset[i]['attention_mask']).item(), max_seq_len) for i in range(len(dataset))]
    length_counts = Counter(lengths)
    length_probs = torch.tensor([length_counts[i]/len(dataset) for i in range(max_seq_len + 1)])
    assert length_probs[0] == 0, 'Can\'t have examples of length 0'

    length_distribution = torch.distributions.Categorical(probs=length_probs)

    return length_distribution
