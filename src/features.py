import os
import csv
import argparse

import numpy as np
import librosa
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# Initialise wav2vec2 processor and model for embedding extraction
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")