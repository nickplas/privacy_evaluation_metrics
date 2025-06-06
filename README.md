# privacy_evaluation_metrics
Code for the paper "Empirical Evaluation of Synthetic Data Privacy Metrics]{Empirical Evaluation of Structured Synthetic
Data Privacy Metrics: Novel framework and application to no-box metrics"

### Installation

```shell
path/to/privacy_eval$ pip install .
```

To use the anonymeter package follow the installation instructions in https://github.com/statice/anonymeter. To install
the embedding method follow the installation instruction in https://github.com/aindo-com/privacy-eval-paper.

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

# Tested Utility Metrics

- [x] Machine Learning Efficacy