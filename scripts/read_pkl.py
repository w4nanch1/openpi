import pickle, numpy as np
path = '/root/openpi/libero_compare/base/group_00000_00099.pkl'
with open(path, 'rb') as f:
    items = pickle.load(f)

print('文件条数:', len(items))
first = items[0]
print('keys:', first.keys())

acts = first['activations']
print('layers:', acts.keys())

for k,v in acts.items():
    print(f'layer {k}: shape {v.shape}, dtype {v.dtype}')

print('expert_actions shape', first['expert_actions'].shape, first['expert_actions'].dtype)
print('prompt_tokens', None if first['prompt_tokens'] is None else (first['prompt_tokens'].shape, first['prompt_tokens'].dtype))


