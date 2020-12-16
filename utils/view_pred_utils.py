import numpy as np
import torch
import shutil
import matplotlib.pyplot as plt
import os
from PIL import Image


def get_pred_from_cls_output(outputs):
    preds = []
    for n in range(0, len(outputs)):
        output = outputs[n]
        _, pred = output.topk(1, 1, True, True)
        preds.append(pred.view(-1))
    return preds