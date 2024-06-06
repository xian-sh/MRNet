from .ret_model import RET
ARCHITECTURES = {"RET": RET}

def build_model(cfg):
    return ARCHITECTURES[cfg.MODEL.ARCHITECTURE](cfg)
