# Results (speechocean762 test set: 2500 utterances, 125 unseen speakers)

## Total score, by model

|                                 |   pearson |   pearson_lo |   pearson_hi |   qwk |   exact_agree |   adjacent_agree |   rmse |
|:--------------------------------|----------:|-------------:|-------------:|------:|--------------:|-----------------:|-------:|
| Mean                            |   nan     |      nan     |      nan     | 0     |         0.232 |            0.684 |  1.551 |
| Speech rate only                |     0.394 |        0.305 |        0.476 | 0.266 |         0.309 |            0.752 |  1.426 |
| Ridge, all features             |     0.68  |        0.615 |        0.732 | 0.64  |         0.408 |            0.846 |  1.144 |
| Gradient boosting, all features |     0.697 |        0.626 |        0.749 | 0.643 |         0.414 |            0.863 |  1.113 |
| Multi-head (shipped)            |     0.676 |        0.613 |        0.728 | 0.636 |         0.404 |            0.846 |  1.147 |

## Shipped model, each head against its own human label

|          |   this model |   GOPT (published) |   one expert vs. others |
|:---------|-------------:|-------------------:|------------------------:|
| accuracy |        0.54  |              0.714 |                   0.732 |
| fluency  |        0.651 |              0.753 |                   0.768 |
| prosodic |        0.623 |              0.76  |                   0.758 |
| total    |        0.676 |              0.742 |                   0.777 |

## Construct validity: each feature group alone (ridge), Pearson on test

|          |   accuracy |   fluency |   prosodic |   total |
|:---------|-----------:|----------:|-----------:|--------:|
| delivery |      0.534 |     0.659 |      0.623 |   0.561 |
| accuracy |      0.546 |     0.533 |      0.549 |   0.557 |
| prosody  |      0.214 |     0.231 |      0.209 |   0.214 |
| all      |      0.66  |     0.728 |      0.714 |   0.682 |

## Subgroups (total score, test set)

|             |    n |   human_mean |   model_mean |   pearson |    smd |
|:------------|-----:|-------------:|-------------:|----------:|-------:|
| adult (16+) | 1220 |        7.04  |        7.375 |     0.823 |  0.216 |
| child (<16) | 1280 |        7.476 |        7.154 |     0.463 | -0.208 |
| f           | 1160 |        7.195 |        7.271 |     0.727 |  0.049 |
| m           | 1340 |        7.322 |        7.253 |     0.63  | -0.045 |

## Combiner weights (total = sum of weight x head score + intercept)

|          |   weight |
|:---------|---------:|
| accuracy |    0.685 |
| fluency  |    0.357 |
| prosodic |    0.364 |

Conformal 90% half-width (from grouped out-of-fold residuals): 1.83. Empirical test coverage: 0.908.
