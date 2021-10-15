# Training scripts

* train.py: unused for now
* dual_train4x.py: networks that output mask & semantics
* mask_train.py: networks that output only mask
* sem_train.py: networks that output only semantics (mask is embedded as a semantic class)
* triple_trian.py: networks that output mask, semantics and directly estimate and supervise noise [WIP]

# Misc scripts
* gui.py: evaluate networks, show ouput, inject arbitrary noise, save samples to disk
* calculate_ce_weights.py: calculate weights for cross entropy (unstable, discontinued)
* generate_wandb_report.py: create a wandb report with hard samples (no longer functional)
* evaluate.py: evaluate trained networks, save samples to disk (merged into gui, discontinued)
