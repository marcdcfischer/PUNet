from torch.utils.data._utils.collate import default_collate


def collate_list(batch):
    """Flattens list and passes it to standard collate function"""

    batch = [item_ for elements_ in batch for item_ in elements_]

    return default_collate(batch)
