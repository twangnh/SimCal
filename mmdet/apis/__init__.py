from .env import get_root_logger, init_dist, set_random_seed
from .inference import (inference_detector, init_detector, show_result,
                        show_result_pyplot)
# from .train import train_detector
from .train_new import train_detector as train_detector_calibration
from .train_orig import train_detector as train_detector_normal
__all__ = [
    'init_dist', 'get_root_logger', 'set_random_seed', 'train_detector_calibration', 'train_detector_normal',
    'init_detector', 'inference_detector', 'show_result', 'show_result_pyplot'
]
