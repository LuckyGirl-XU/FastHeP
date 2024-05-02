# TecHNet

This repo is the open-sourced code for the proposed TecHNet

# Requirements

- python >= 3.6.13
- pandas==1.4.3
- tqdm==4.41.1
- numpy==1.23.1
- scikit_learn==1.1.2
- pytorch >= 1.8.1
- dgl >= 0.6.1


# Datasets
The datasets are used in this paper:

- We collect eight datasets from (https://www.cs.cornell.edu/~arb/data/).

# Clone the repository:

    git clone https://github.com/Graph-COM/Neighborhood-Aware-Temporal-Network
    cd TecHNet

# Preprocessing
  ## Temporal sampler initialization

    python utils/setup.py build_ext --inplace

  ## Data preprocessing
  - We provide preprocessed data samples, which can be downloaded [here](https://drive.google.com/drive/folders/1Nr9bL6rEkioR9gzftEPP3fk4J7pLodLs?usp=sharing)
  - Raw data need to be processing:
```
  python utils/gen_graph.py --data WIKI
```



# Run Examples
  ## Single GPU training: Link prediction task and Link ranking task
  
  
      - For transductive link prediction
          python train.py --data WIKI --config ./config/TimeSGN.yml --gpu 0 --DTMP
      - For inductive link Ranking
          python train.py --data WIKI --config ./config/TimeSGN.yml --gpu 0 --eval_can_samples 100 --DTMP --use_inductive 

 
  ## Multi-GPU training for billion-scale datasets
      
      - For transductive link prediction
          python -m torch.distributed.launch --nproc_per_node=9 train_dist.py --data GDELT --config ./config/dist/TimeSGN.yml --num_gpus 8 
