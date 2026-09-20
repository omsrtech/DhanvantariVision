# Data and attribution

**No radiographs are distributed in this repository.** The code, the derived
metrics and the trained weights are ours and are Apache-2.0. The images are
not ours to pass on, so `scripts/fetch_*.py` and `scripts/prepare.py` rebuild
the data locally from the original providers instead.

## Sources

### TB model — NLM Montgomery County & Shenzhen
800 radiographs: 138 from the Department of Health and Human Services,
Montgomery County, Maryland, USA; 662 from Shenzhen No. 3 People's Hospital,
Guangdong, China. Released by the U.S. National Library of Medicine, National
Institutes of Health, Bethesda, MD, USA.

The NLM asks that recipients **not redistribute these datasets** and that new
users request them from the provider directly, so they are excluded here.

Request access: <http://archive.nlm.nih.gov/repos/chestImages.php>

Cite:
> Jaeger S, Candemir S, Antani S, Wáng YX, Lu PX, Thoma G. *Two public chest
> X-ray datasets for computer-aided screening of pulmonary diseases.*
> Quant Imaging Med Surg. 2014;4(6):475-477.

> Candemir S, Jaeger S, Palaniappan K, et al. *Lung segmentation in chest
> radiographs using anatomical atlases with nonrigid registration.*
> IEEE Trans Med Imaging. 2014;33(2):577-590.

### Nodule / mass model — NIH ChestX-ray14
2,214 films from 1,843 patients, split by patient. Provided by the NIH
Clinical Center.

Cite:
> Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM. *ChestX-ray8:
> Hospital-scale chest X-ray database and benchmarks on weakly-supervised
> classification and localization of common thorax diseases.* CVPR 2017.

### Backbone
ResNet-18 initialised from ImageNet weights. ImageNet's own terms apply to
that initialisation; the fine-tuned checkpoints released here are our work.

## What *is* in this repository

`data/*/predictions.csv` and `data/*/labels.csv` are derived tables — image
id, split, label, model probability, decision, Grad-CAM zone fractions. They
contain no pixel data and no patient information. `reports/` holds the
metrics and figures computed from them.

## Not a medical device

Research prototype. Not cleared by any regulator, not clinically validated,
and not for decisions about any real patient.
