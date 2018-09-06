# Clintox results
(Full tables in /etc/results/)
### Top-10
#### Random
| Model | Descriptors                             | n_bits | ROC AUC         | ROC AUC val     |
|-------|-----------------------------------------|--------|-----------------|-----------------|
| xgb   | ['rdkit', 'morgan', 'mordred', 'maccs'] | 1024   | 0.94845±0.00455 | 0.88955±0.00935 |
| xgb   | ['rdkit', 'maccs']                      |        | 0.92540±0.00490 | 0.91605±0.02535 |
| xgb   | ['mordred', 'maccs']                    |        | 0.92375±0.02215 | 0.90255±0.02665 |
| svc   | ['rdkit', 'maccs']                      |        | 0.92085±0.00035 | 0.89220±0.00000 |
| xgb   | ['rdkit', 'morgan', 'mordred', 'maccs'] | 64     | 0.91815±0.01135 | 0.91210±0.03980 |
| xgb   | ['morgan', 'maccs']                     | 1024   | 0.91695±0.01015 | 0.87445±0.03205 |
| xgb   | ['rdkit', 'morgan', 'mordred', 'maccs'] | 512    | 0.90905±0.02045 | 0.91265±0.04075 |
| svc   | ['morgan', 'maccs']                     | 32     | 0.90610±0.00000 | 0.86625±0.00075 |
| xgb   | ['morgan', 'maccs']                     | 256    | 0.90570±0.00110 | 0.88840±0.02180 |
| xgb   | ['morgan', 'maccs']                     | 512    | 0.90160±0.00900 | 0.88655±0.01955 |

#### Cluster
| Model | Descriptors                             | n_bits | ROC AUC         | ROC AUC val     |
|-------|-----------------------------------------|--------|-----------------|-----------------|
| xgb   | ['rdkit', 'morgan', 'mordred', 'maccs'] | 256    | 0.96845±0.00045 | 0.91615±0.00155 |
| xgb   | ['rdkit', 'morgan', 'mordred', 'maccs'] | 64     | 0.96735±0.00035 | 0.91145±0.00155 |
| xgb   | ['rdkit', 'morgan', 'mordred', 'maccs'] | 2048   | 0.96680±0.00110 | 0.92130±0.00270 |
| xgb   | ['rdkit', 'morgan', 'mordred', 'maccs'] | 512    | 0.96670±0.00070 | 0.91280±0.00520 |
| xgb   | ['rdkit', 'morgan', 'mordred', 'maccs'] | 1024   | 0.96500±0.00170 | 0.92450±0.00230 |
| xgb   | ['mordred', 'maccs']                    |        | 0.96140±0.00170 | 0.89900±0.00080 |
| xgb   | ['rdkit', 'maccs']                      |        | 0.95780±0.00770 | 0.90815±0.00055 |
| xgb   | ['morgan', 'maccs']                     | 32     | 0.95660±0.00000 | 0.88070±0.00000 |
| xgb   | ['maccs']                               |        | 0.95420±0.01210 | 0.85835±0.00985 |
| xgb   | ['morgan', 'maccs']                     | 256    | 0.94660±0.01410 | 0.88130±0.01630 |

#### Scaffold
| Model | Descriptors                             | n_bits | ROC AUC         | ROC AUC val     |
|-------|-----------------------------------------|--------|-----------------|-----------------|
| xgb   | ['rdkit', 'morgan','mordred', 'maccs']  | 1024   | 0.93630±0.00000 | 0.85270±0.00000 |
| xgb   | ['rdkit', 'morgan','mordred', 'maccs']  | 2048   | 0.93380±0.00000 | 0.87600±0.00000 |
| xgb   | ['rdkit', 'morgan','mordred', 'maccs']  | 512    | 0.92690±0.00000 | 0.85600±0.00000 |
| xgb   | ['rdkit', 'morgan','mordred', 'maccs']  | 32     | 0.92680±0.00000 | 0.95780±0.00000 |
| xgb   | ['rdkit', 'morgan','mordred', 'maccs']  | 256    | 0.91800±0.00000 | 0.86490±0.00000 |
| xgb   | ['rdkit', 'morgan','mordred', 'maccs']  | 32     | 0.91800±0.00000 | 0.85990±0.00000 |
| xgb   | ['rdkit', 'morgan','mordred', 'maccs']  | 64     | 0.91200±0.00000 | 0.86380±0.00000 |
| xgb   | ['maccs']                               |        | 0.90560±0.00000 | 0.84510±0.00000 |
| xgb   | ['mordred', 'maccs']                    |        | 0.90180±0.00000 | 0.85660±0.00000 |
| xgb   | ['rdkit', 'morgan', 'mordred', 'maccs'] | 2048   | 0.90100±0.00000 | 0.88910±0.00000 |


### Top-10 models
#### Random
| Model | ROC AUC         | ROC AUC val     |
|-------|-----------------|-----------------|
| xgb   | 0.82908±0.16620 | 0.82038±0.16950 |
| svc   | 0.82439±0.13715 | 0.86918±0.19355 |
| rf    | 0.77849±0.12020 | 0.73903±0.12845 |
| lr    | 0.77223±0.11775 | 0.71021±0.19420 |
| knn   | 0.64659±0.12230 | 0.67728±0.12915 |

#### Cluster
| Model | ROC AUC         | ROC AUC val     |
|-------|-----------------|-----------------|
| xgb   | 0.82386±0.19440 | 0.78404±0.16845 |
| svc   | 0.82057±0.17625 | 0.76584±0.14655 |
| lr    | 0.79265±0.12540 | 0.74828±0.07915 |
| rf    | 0.75685±0.12965 | 0.71603±0.13965 |
| knn   | 0.60614±0.12090 | 0.64342±0.08475 |

#### Scaffold
| Model | ROC AUC         | ROC AUC val     |
|-------|-----------------|-----------------|
| xgb   | 0.83954±0.20450 | 0.84868±0.25165 |
| rf    | 0.78191±0.13860 | 0.73206±0.20955 |
| lr    | 0.77488±0.16585 | 0.83096±0.18400 |
| svc   | 0.75635±0.14910 | 0.82142±0.18875 |
| knn   | 0.58136±0.12635 | 0.62689±0.22460 |


### Top-10 descriptors
#### Random
| Descriptors                             | n_bits | ROC AUC         | ROC AUC val     |
|-----------------------------------------|--------|-----------------|-----------------|
| ['rdkit', 'maccs']                      |        | 0.82543±0.14450 | 0.79530±0.15685 |
| ['rdkit', 'morgan', 'mordred', 'maccs'] | 1024   | 0.82507±0.13105 | 0.80894±0.14650 |
| ['rdkit', 'morgan', 'mordred', 'maccs'] | 256    | 0.81999±0.12160 | 0.83475±0.09315 |
| ['morgan', 'maccs']                     | 32     | 0.81795±0.13470 | 0.75235±0.15405 |
| ['morgan', 'maccs']                     | 256    | 0.81581±0.13260 | 0.75220±0.15070 |
| ['morgan', 'maccs']                     | 64     | 0.81502±0.12715 | 0.74733±0.14830 |
| ['morgan', 'maccs']                     | 512    | 0.81437±0.12940 | 0.75510±0.14865 |
| ['morgan', 'maccs']                     | 1024   | 0.81360±0.13955 | 0.74352±0.17420 |
| ['rdkit', 'morgan', 'mordred', 'maccs'] | 64     | 0.81349±0.11700 | 0.81450±0.15700 |
| ['maccs']                               |        | 0.81135±0.13430 | 0.74559±0.13170 |

#### Cluster
| Descriptors                             | n_bits | ROC AUC         | ROC AUC val     |
|-----------------------------------------|--------|-----------------|-----------------|
| ['rdkit', 'morgan', 'mordred', 'maccs'] | 512    | 0.82301±0.17495 | 0.77768±0.11285 |
| ['rdkit', 'morgan', 'mordred', 'maccs'] | 2048   | 0.82250±0.15880 | 0.77054±0.12360 |
| ['rdkit', 'morgan', 'mordred', 'maccs'] | 1024   | 0.82140±0.17295 | 0.77798±0.12635 |
| ['rdkit', 'morgan', 'mordred', 'maccs'] | 64     | 0.81992±0.18620 | 0.77256±0.14630 |
| ['rdkit', 'morgan', 'mordred', 'maccs'] | 256    | 0.81945±0.17505 | 0.77780±0.12155 |
| ['maccs']                               |        | 0.81093±0.20560 | 0.77676±0.07750 |
| ['mordred', 'maccs']                    |        | 0.80821±0.18355 | 0.76299±0.14035 |
| ['rdkit', 'maccs']                      |        | 0.80576±0.21225 | 0.77799±0.12805 |
| ['morgan', 'maccs']                     | 1024   | 0.80344±0.17065 | 0.75547±0.13410 |
| ['morgan', 'maccs']                     | 32     | 0.79671±0.20850 | 0.76345±0.11340 |

#### Scaffold
| Descriptors                            | n_bits | ROC AUC         | ROC AUC val     |
|----------------------------------------|--------|-----------------|-----------------|
| ['maccs']                              |        | 0.90560±0.00000 | 0.84510±0.00000 |
| ['rdkit', 'morgan','mordred', 'maccs'] | 2048   | 0.88460±0.04920 | 0.83280±0.04320 |
| ['rdkit', 'maccs']                     |        | 0.87105±0.00595 | 0.80840±0.06200 |
| ['rdkit', 'maccs']                     |        | 0.86570±0.00000 | 0.79180±0.00000 |
| ['rdkit']                              |        | 0.85500±0.00000 | 0.77740±0.00000 |
| ['rdkit']                              |        | 0.84585±0.02305 | 0.76465±0.01275 |
| ['rdkit', 'mordred']                   |        | 0.84490±0.00000 | 0.84940±0.00000 |
| ['mordred']                            |        | 0.81550±0.00000 | 0.82120±0.00000 |
| ['rdkit', 'morgan','mordred', 'maccs'] | 512    | 0.81380±0.12755 | 0.75362±0.12815 |
| ['rdkit', 'morgan','mordred', 'maccs'] | 1024   | 0.81288±0.09710 | 0.77493±0.09525 |

### Plots
- [Random](../preprocessed_clintox_random/clintox_random_results.md)
- [Cluster](../preprocessed_clintox_cluster/clintox_cluster_results.md)
- [Scaffold](../preprocessed_clintox_scaffold/clintox_scaffold_results.md)
- [Random val](../preprocessed_clintox_random/clintox_random_results_val.md)
- [Cluster val](../preprocessed_clintox_cluster/clintox_cluster_results_val.md)
- [Scaffold val](../preprocessed_clintox_scaffold/clintox_scaffold_results_val.md)