from models import ECAPA_TDNN
import torch


model = ECAPA_TDNN()
model.load_state_dict(torch.load('./pre_trained/ECAPA_TDNN.model'))