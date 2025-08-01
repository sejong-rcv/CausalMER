import torch
import os
from src.dataset import Multimodal_Datasets

def get_data(args, dataset, split='train'):
    alignment = 'a' if args.aligned else 'na'
    data_path = os.path.join(args.data_path, dataset) + f'_{split}_{alignment}.dt'
    if not os.path.exists(data_path):
        print(f"  - Creating new {split} data")
        data = Multimodal_Datasets(args.data_path, dataset, split, args.aligned)
        torch.save(data, data_path)
    else:
        print(f"  - Found cached {split} data")
        data = torch.load(data_path) # data # <src.dataset.Multimodal_Datasets object at 0x7f0d63ff8f10>
    return data


def save_load_name(args, name=''):
    if args.aligned:
        name = name if len(name) > 0 else 'aligned_model'
    elif not args.aligned:
        name = name if len(name) > 0 else 'nonaligned_model'

    return name + '_' + args.model


def save_model(args, model, name=''):
    name = save_load_name(args, name)
    torch.save(model, f'checkpoint({args.dataset})/{name}.pt')

def load_model(args, name=''):
    name = save_load_name(args, name)
    model = torch.load(f'checkpoint({args.dataset})/{name}.pt')
    return model

def padTensor(t: torch.tensor, targetLen: int) -> torch.tensor:
    oriLen, dim = t.size()
    return torch.cat((t, torch.zeros(targetLen - oriLen, dim).to(t.device)), dim=0)