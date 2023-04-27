"""
Implementation of CLIP and its image/text models in Flax.
"""


from .factory import create_model, list_models
from .loss import CLIPLoss, CLIPWithLoss
from .model import CLIP
