# privacy_evaluation_metrics

A python package for evaluating privacy and utility for single-table datasets for the [paper "EMPIRICAL EVALUATION OF STRUCTURED SYNTHETIC DATA PRIVACY METRICS: NOVEL EXPERIMENTAL FRAMEWORK"](https://arxiv.org/pdf/2512.16284) by Milton Nicolás Plasencia Palacios, Alexander Boudewijn, Sebastiano Saccani, Andrea Filippo Ferraris, Diana Sofronieva, Giuseppe D’Acquisto Filiberto Brozzetti, Daniele Panfilo, and Luca Bortolussi.

### Installation

```shell
path/to/privacy_eval$ pip install .
```

To use the anonymeter package follow the installation instructions in [link to anonymeter](https://github.com/statice/anonymeter). To install
the embedding method follow the installation instruction in [link to contrastive learning method](https://github.com/aindo-com/privacy-eval-paper).

---

The `examples` folder contain the tutorials to run the privacy and utility metrics. In the same folder, 
`outlier_removal.ipynb` shows how to remove the outliers from a dataset. The tutorials are meant to be run with the 
Adult dataset that can be downloaded from [here](https://archive.ics.uci.edu/dataset/2/adult).

# Tested Privacy Metrics

- [x] IMS
- [x] DCR
- [x] MIA
- [x] Singling Out
- [x] Linkability
- [x] Inference
- [x] Machine Learning Inference
- [x] GTCAP
- [x] DOMIAS

### DOMIAS Issue
Most of the times Domias may incur in a Segmentation Fault. Set the number of threads to 1 to make it work. 

# Tested Utility Metrics

- [x] Machine Learning Efficacy
- [x] Regression and Classification on a subset
- [x] K-S test
- [x] Chi-Squared Test
- [x] Maximum Mean Discrepancy
