from models.ResNet34_DSBN import ResNet34_DSBN
import torch 
import re

d = torch.normal(mean = 0, std = 1, size = (1, 16000))
model = ResNet34_DSBN(nOut = 512, encoder_type = 'ASP').to('cpu')
model.load_state_dict(torch.load('./pre_trained/resnet34_DSBN.model'))
model(d, 'source')

# state_dict = torch.load('./pre_trained/resnet34.model', map_location='cpu')
# new_dict = {}
# for key, val in state_dict.items():
    
#     if 'softmax' in key or 'angleproto' in key:
#         continue
    
#     if '__S__' in key:
#         key = key.replace("__S__.", '')
#     if '__L__' in key: 
#         key = key.replace('__L__.', '')
        
#     if 'bn' in key:
#         bn = re.findall(r'(bn\d+)', key)
#         if bn:
#             bn = bn[0]
#             new_dict[key.replace(bn, f'{bn}.bn_source')] = val
#             new_dict[key.replace(bn, f'{bn}.bn_target')] = val
#         else:
#             new_dict[key.replace('bn_last', f'bn_last.bn_source')] = val
#             new_dict[key.replace('bn_last', f'bn_last.bn_target')] = val
            
#     elif 'downsample.0' in key:
#             new_dict[key.replace('downsample.0', f'downsample.conv2d')] = val
#     elif 'downsample.1' in key:
#             new_dict[key.replace('downsample.1', f'downsample.bn.bn_source')] = val
#             new_dict[key.replace('downsample.1', f'downsample.bn.bn_target')] = val
#     elif 'attention.0' in key:
#             new_dict[key.replace('attention.0', f'attention.conv1')] = val
#     elif 'attention.2' in key:
#             new_dict[key.replace('attention.2', f'attention.bn.bn_source')] = val
#             new_dict[key.replace('attention.2', f'attention.bn.bn_target')] = val
#     elif 'attention.3' in key:
#             new_dict[key.replace('attention.3', f'attention.conv2')] = val
#     else:
#         new_dict[key] = val
    
        
# torch.save(new_dict, './pre_trained/resnet34_DSBN.model')
# model.load_state_dict(new_dict)
 