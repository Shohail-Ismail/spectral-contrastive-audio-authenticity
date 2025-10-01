# Spectral Contrastive Audio Authenticity
Audio deepfake detection with novel spectral-contrastive loss.


## Data Acquisition
### FakeAVCeleb
https://www.kaggle.com/datasets/sardertanvirahmed/fakeavcaleb -> data/raw/fakeav

### ASVspoof2019 LA dev-set
https://datashare.ed.ac.uk/handle/10283/3336 -> LA.zip -> data/rawasvspoof

---

## Implementation

## Header:
**file** = ID from asvspoof2019 LA dev set (e.g. LA_0001)

**label** = 0 = real human speech , 1 = spoofed audio (as specified in `ASVspoof2019.LA.cm.dev.trl.txt`) 

**centroid** = Mean spectral centroid (timbral sharpness - mean audio brightness)

**entropy** = entropy over mel-spectrogram bins (signal complexity)

**mfcc_0...mfcc_12** = average mfcc (13 values)

**emb_0...emb_767** = wav2vec2 embedding (768-dim vector)

## Parallelism

24844 files so parallelism used to speed things up - done before creating training script

For `preprocess_datasets.py`: reduced time from estimated 8 hours to 40 mins on 11 cores (estimate = time taken for 100 * 2500 (25000 files))

For `features.py`: 1 core takes estimated 8 hours so further optimisations needed -> trialling further optimisations only slowed process down due to inference dominating runtime, some preprocessing still running serially, and padding to the longest clip negating batching/quantisation gains -> decided to run feature extraction overnight as efficiency in this step is not critical to core research.

Final iteration could be optimised by refactoring preprocess_datasets.py to absorb file-loading and numpy work from features.py -> decreases time drastically

## Baseline training

### train_test_split
Used to ensure random shuffling before split to avoid bias (reproducibility included, seed = 10)

### Feature scaling
Applied to training and test to bring centroid, mfccs and embeddings onto a common scale for better logreg convergence

### 5 fold CV grid search
C vals = 0.01, 0.1, 1.0 using GridSearchCV so mean ROC AUC across 5 folds to give best C value. This optimal C is then used to refit model on full training set.

## Contrastive training

Frozen wav2vec to skip backprop over 300M params (takes estimated training time from ~40 hours to ~20). May gain statistically significant AUC points so shelf that for later when GPU => good way might be to:
    - run every file through, fit a logreg classifier and report roc-auc
    - in second run, unfreeze last 1 or 2 transformer layers and fine tune those + the 128-dim projection head -> compare roc auc to see if >=1% change at least from baseline 0.9744

---

### Reminder of why's and end-goals for when project starts up again (guessing briefly in Christmas break or when the other 2 side projects alongside work end in Nov-ish - let me (you) know if you (me) are right!):

**Why did I/you start this project?**
Get some XP in what research in ML looks like, and some knowledge in ML in general. Also, audio deepfakes are a misinfo issue, so thats an aligment there - wanted something real to solve without AI-ing world peace or sommet. Audio is super interesting and logical, and problem-solving (like parallelising) is nice. Fun lessons. And now I know why efficiency is important in research lol.

**Papers from research**
- [Exploring Green AI for Audio Deepfake Detection](https://arxiv.org/pdf/2403.14290)
    - Green AI = small performance -ve, big environment +ve - most spoof detection is Red AI = small performance +ves (Accuracy) w very high env cost.
    - No need to fine-tune giant models thoug. Just take embeddings from a pretrained wav2vec2 model, avg them, and feed into light classifiers like loggreg or SVM. Run it all on CPU, and you still get rly strong results (down to 0.9% equal error rate)
- [Towards Attention-based Contrastive Learning for Audio Spoof Detection](https://arxiv.org/html/2407.03514v1)
    - Link is messed up but it shows contrastive + attention in audio spoof detection works. Good reference point, and tbh a mini cross-attention module may help for the whole lightweight aspect.
    - 3 branch Siamese network (2 self attention + 1 cross-attention that merges info from both) - more info within but their model is probably heavier than the CPU-only target. We can just scale down their insights though.
- Will add the rest tomorrow. Need sleep. blegh.


- **End-goals:**
1) Implementing the actual spectral-contrastive part, including "CPU-only". Remember the research to arrive here.
    - Lightweight aspect here is primarily because an audio-deepfake-detection app or browser extension would be super helpful. Also, less compute = less planet destruction. Main paper is 'Exploring Green AI for Audio Deepfake Detection' above.
    - ASVspoof 2019 paper: (paraphrasing) most submitted systems rely on spectrogram-based features because spoofs leave weird frequency artefacts, seen through spectral patterns.
    - Contrastive training to fit with the lightweight part but also it just fits naturally with comparing different spectral views of the same audio. Papers above demonstrate GAN discriminators and full fine-tuning are 5x more accurate but 20x more polluting ([source for figures ( ͡° ͜ʖ ͡°)](https://en.wikipedia.org/wiki/Hyperbole)) 

2) WB2/EWB/TempestExtremes benchmarking has given a good idea so far of how/what/why benchmarking in AI research, and Aurora is easy enough to understand.We had baseline persistence ERA5 vs Aurora for that, whereas baselines here are ROC-AUC, EER, AP, calibration curves and ECE, and report thresholds chosen on validation while also giving threshold-free metrics, with 95% confidence intervals from bootstrap resampling. Lot of big words, open a couple tabs you'll get it in no time. Muscle memory and all that.

3) Testing skills are v important => ablation tests. Potentials are removing the contrastive loss namesake, diff augmentations, diff head sizes, and compare freezing strategies to isolate the effects of spect-contra (this has not been done already).
→ Ablations section. Necessary because without ablations, nobody can tell whether performance gains are due to your proposed method or just due to extra capacity/regularisation choices.

4) Remember again, this is meant to be lightweight so the end demo pipeline/notebook/youtube video/idk must output latency, ram, model size, decision scores, and a overlay of predictions on spectrograms.

5) Doug (Data-Driven Climate Modelling senior research scientist, Jun-Sep manager) mentioned reproducibility remember. So include a bunch of seeds for randoms, and mone-shot reruns via make commands (?). He also mentioned a 'model card' with potential side-uses, training data/biases, eval conditions, and limitations. This is for AI ethics and safe deployment, so it is absolutely necessary.

6) A bunch of other stuff (interpretability, etc.)

5) Nice-looking README.