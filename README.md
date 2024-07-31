# FastHeP

This repo is the open-sourced code for the proposed FastHeP

# Requirements

    - python >= 3.6.13
    - pandas==1.4.3
    - tqdm==4.41.1
    - numpy==1.23.1
    - scikit_learn==1.1.2
    - pytorch >= 1.8.1
    - dgl >= 0.6.1

# Datasets
   We collect the following eight datasets from https://www.cs.cornell.edu/~arb/data/. 
   
      - drug networks
        NDC-classes, NDC-Substances
      - the network of congress cosponsoring bills
        Congress bills
      - email networks 
        Email-Enron
      - social networks 
        Users-Threads, Threads-Math-Sx, Threads-Stack-Overflow
      - co-authorship on DBLP 
        Coauth-DBLP
            
   These real-world datasets can be downloaded from [here](https://www.cs.cornell.edu/~arb/data/) to HG_Data/. Then run the following: 

     cd HG_Data/
     unzip [dataset_name].zip

     
# Run Examples
  ## Hyperedge prediction in transductive setting
  
    python main.py -d email-Enron --bs 100 --n_degree 16 --n_hop 1 --mode t --replace_prob 0.8 --gpu 1 --run 1 --split 0.7  
  
  ## Hyperedge prediction in strongly and weakly inductive settings
  
    python main.py -d email-Enron --bs 100 --n_degree 16 --n_hop 1 --mode i --replace_prob 0.8 --gpu 1 --run 1 --split 0.7  

