from dataclasses import dataclass

@dataclass
class EvoConfig:
    drop_attn: bool = False
    denoise_e2e: bool = False