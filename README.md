# spectral-contrastive-audio-authenticity
Audio deepfake detection with novel spectral-contrastive loss


# data acquisition
## FakeAVCeleb
https://www.kaggle.com/datasets/sardertanvirahmed/fakeavcaleb -> data/raw/fakeav

## ASVspoof2019 LA dev-set
https://datashare.ed.ac.uk/handle/10283/3336 -> LA.zip -> data/rawasvspoof


### Header:
**file** = ID from asvspoof2019 LA dev set (e.g. LA_0001)

**label** = 0 = real human speech , 1 = spoofed audio (as specified in `ASVspoof2019.LA.cm.dev.trl.txt`) 

**centroid** = Mean spectral centroid (timbral sharpness - mean audio brightness)

**entropy** = entropy over mel-spectrogram bins (signal complexity)

**mfcc_0...mfcc_12** = average mfcc (13 values)

**emb_0...emb_767** = wav2vec2 embedding (768-dim vector)