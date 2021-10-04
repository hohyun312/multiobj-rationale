# Multi-Objective Molecule Generation using Interpretable Substructures

This is the implementation of ICML 2020 paper: https://arxiv.org/abs/2002.03244

You can see original implementation in https://github.com/wengong-jin/multiobj-rationale



# Requirements

Lower version of python (<=3.7) is required to use pre-trained random forest model defined in sklearn. `networkx`, `pytorch` and `rdkit` is also necessary.

```
conda create -c conda-forge -n my-rdkit-env rdkit python=3.7
```



# Data Extraction

For downloading ExCAPE-DB, please go to `data` directory and run `get_excape.py`. Or you can directly use files in `data/excape-db` as it is already processed.

```
python get_excape.py
```



## Rationale Extraction

The following command will extract rationales using MCTS and produce two files in `data` directory: `jnk3_rationale.csv`, `gsk3b_rationale.csv`.

```
python mcts.py --data jnk3
python mcts.py --data gsk3b
```

To construct multi-property rationales, run the following.

```
python merge_rationale.py
```

