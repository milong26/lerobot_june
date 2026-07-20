import torch

import transformers

from typing import Dict, Optional, Sequence, List


def find_all_linear_names(model, rank0_print, lora_module=None):
    cls = torch.nn.Linear
    lora_module_names = set()
    lang_type = 'phi' if 'phi' in model.name_or_path.lower() else 'pythia'
    multimodal_keywords = ['vision_resampler', 'mm_projector', 'embed_out', 'proj_to_action']
    if 'vit' not in lora_module:
        multimodal_keywords.append("vision_tower")
    rank0_print("##" * 20)

    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue

        if lang_type == 'pythia':
            if ('embed_out' not in name) and ('llm' not in lora_module) and ('layers' in name) and ('vision' not in name) and ('gpt_neox' in name):
                continue

        elif lang_type == 'phi':
            if ('embed_out' not in name) and ('llm' not in lora_module) and ('layers' in name) and ('vision' not in name) and ('model' in name):
                continue

        if isinstance(module, cls):
            lora_module_names.add(name)

    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')

    if 'half' in lora_module:
        new_lora_module_names = set()
        for n in lora_module_names:
            if ('embed_out' not in n) and ('layers' in n) and ('vision' not in name) and ('gpt_neox' in n):
                if int(n.split('.')[2]) % 2 == 0:
                    continue
                else:
                    new_lora_module_names.add(n)
            else:
                new_lora_module_names.add(n)
        lora_module_names = new_lora_module_names

    return list(lora_module_names)