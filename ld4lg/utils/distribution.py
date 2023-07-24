from collections import Counter

import torch


def get_length_distribution(dataset, max_seq_len, mask_key='attention_mask'):
    """
    Given a dataset, compute a length distribution for the sequence lengths found in the dataset.

    Args:
        dataset (datasets.arrow_dataset.Dataset): The dataset to analyze.
        max_seq_len (int): The maximum sequence length considered.
        mask_key (str, optional): The key used to access the mask in the dataset's examples.
            Defaults to 'attention_mask'.

    Returns:
        length_distribution (torch.distributions.Categorical): A categorical distribution of sequence lengths.
    """

    lengths = [min(sum(dataset[i][mask_key]).item(), max_seq_len) for i in range(len(dataset))]
    length_counts = Counter(lengths)
    length_probs = torch.tensor([length_counts[i] / len(dataset) for i in range(max_seq_len + 1)])
    assert length_probs[0] == 0, 'Can not have examples of length 0'

    length_distribution = torch.distributions.Categorical(probs=length_probs)

    return length_distribution


def get_class_distribution(dataset, class_key='label'):
    """
    Given a dataset, compute a distribution for the classes found in the dataset.

    Args:
        dataset (datasets.arrow_dataset.Dataset): The dataset to analyze.
        class_key (str, optional): The key used to access the class label in the dataset's examples.
            Defaults to 'label'.

    Returns:
        class_distribution (torch.distributions.Categorical): A categorical distribution of class labels.
    """
    classes = [dataset[i][class_key].item() for i in range(len(dataset))]
    class_counts = Counter(classes)
    class_probs = torch.tensor([class_counts[i] / len(dataset) for i in range(len(class_counts))])

    class_distribution = torch.distributions.Categorical(probs=class_probs)

    return class_distribution
