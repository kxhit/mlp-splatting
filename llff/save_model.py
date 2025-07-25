import torch
from model import NeuralPrimitiveScene

def save_model(scene, path="scene_ckpt.pt"):
    torch.save(scene.state_dict(), path)