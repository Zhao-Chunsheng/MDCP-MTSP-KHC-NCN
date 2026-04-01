# Introduction
This project proposes an effective two-phase optimization algorithm, k-means++ hierarchical clustering with neural combinatorial networks (KHC-NCN), tailored for a variant of MTSP: the multi-depot closed-path MTSP (MDCP-MTSP).

# Run
python solving_MTSP_one_instance.py -d data/a280.tsp -k 2 3 4 5 -r 6789


optional arguments:
    -d DATA_FILE, --data_file DATA_FILE: Path to the input TSP data file;
    -k K_LIST [K_LIST ...], --k_list K_LIST [K_LIST ...]: Number of salesmen (e.g., 3 5 8 10);
    -r RANDOM_SEED, --random_seed RANDOM_SEED: Random seed for reproducibility (default: None, using system time ).

# Acknowledgement
We would like to express our gratitude to Kool et al. for their original source code, which has been an essential foundation for constructing and implementing this project. This work would not have been developed without their contributions.
You can find Kool's original source code at: [https://github.com/wouterkool/attention-learn-to-route]
