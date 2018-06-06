# Datasets

A csv format file is required, in which one of the headers must be "smiles", and the rest - the names of the experiments (targets). The column "mol_id" will be dropped if exist.

### BACE
link: http://moleculenet.ai/datasets-1/

Class | Activity
--- | ---
Inactive | 822
Active | 691

<img src="../etc/img/bace/['rdkit', 'morgan', 'mordred', 'maccs']/tsne/t-SNE_split_scaffold.png" /><br />
<img src="../etc/img/bace/['rdkit', 'morgan', 'mordred', 'maccs']/tsne/t-SNE_split_random.png" /><br />
<img src="../etc/img/bace/['rdkit', 'morgan', 'mordred', 'maccs']/tsne/t-SNE_split_stratified.png" /><br />
<img src="../etc/img/bace/['rdkit', 'morgan', 'mordred', 'maccs']/tsne/t-SNE_split_cluster.png" />

### Tox21
link: https://tripod.nih.gov/tox21/challenge/data.jsp

Set | Class | AR	| AR-LBD	| AhR	| Aromatase	| ER	| ER-LBD	| PPAR-g | ARE	 | ATAD5	| HSE	 | MMP | p53
 --- | --- | --- |  --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Train | Inactive | 7197	|	6702	|	5948	|	5669	|	5631	|	6818	|	6422	|	5015	|	7003	|	6260	|	5018	|	6511
Train | Active | 270	|	224	|	767	|	296	|	702	|	319	|	184	|	943	|	252	|	356	|	922	|	419
Test | Inactive | 574 | 574 | 537 | 489 | 465 | 580 | 574 | 462 | 584 | 588 | 483 | 575
Test | Active | 12  | 8   | 73  | 39  | 51  | 20  | 31  | 93  | 38  | 22  | 60  | 41

<img src="../etc/img/tox21/['rdkit', 'morgan', 'mordred', 'maccs']/tsne/t-SNE_split_scaffold.png" /><br />
<img src="../etc/img/tox21/['rdkit', 'morgan', 'mordred', 'maccs']/tsne/t-SNE_split_random.png" /><br />
<img src="../etc/img/tox21/['rdkit', 'morgan', 'mordred', 'maccs']/tsne/t-SNE_split_stratified.png" /><br />
<img src="../etc/img/tox21/['rdkit', 'morgan', 'mordred', 'maccs']/tsne/t-SNE_split_cluster.png" />

### Clintox
link: http://moleculenet.ai/datasets-1/

Class | Activity
--- | ---
Inactive | 1372
Active | 112

<img src="../etc/img/clintox/['rdkit', 'morgan', 'mordred', 'maccs']/tsne/t-SNE_split_scaffold.png" /><br />
<img src="../etc/img/clintox/['rdkit', 'morgan', 'mordred', 'maccs']/tsne/t-SNE_split_random.png" /><br />
<img src="../etc/img/clintox/['rdkit', 'morgan', 'mordred', 'maccs']/tsne/t-SNE_split_stratified.png" /><br />
<img src="../etc/img/clintox/['rdkit', 'morgan', 'mordred', 'maccs']/tsne/t-SNE_split_cluster.png" />
