import torch
from torch.utils.model_zoo import load_url
from PIL import Image
import matplotlib.pyplot as plt

from model.blazeface import FaceExtractor, BlazeFace
from model.architectures import fornet,weights
from model.isplutils import utils

"""
Choose an architecture between
- EfficientNetB4
- EfficientNetB4ST
- EfficientNetAutoAttB4
- EfficientNetAutoAttB4ST
- Xception
"""
net_model = 'EfficientNetAutoAttB4'

"""
Choose a training dataset between
- DFDC
- FFPP
"""
train_db = 'DFDC'

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
face_policy = 'scale'
face_size = 224

model_url = weights.weight_url['{:s}_{:s}'.format(net_model,train_db)]
DEEP_NET = getattr(fornet,net_model)().eval().to(device)
DEEP_NET.load_state_dict(load_url(model_url,map_location=device,check_hash=True))
DEEPFAKE_TRANSFORMER = utils.get_transformer(face_policy, face_size, DEEP_NET.get_normalizer(), train=False)

facedet = BlazeFace().to(device)
facedet.load_weights("./model/blazeface/blazeface.pth")
facedet.load_anchors("./model/blazeface/anchors.npy")
FACE_EXTRACTOR= FaceExtractor(facedet=facedet)

