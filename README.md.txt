Use arguments: <batch_size> <use_batch_norm> <epochs_per_run> <learning_rate_start> <no_of_runs>

positional arguments:
  batch_size           (int) batch size to be used
  use_batch_norm       (int -> 0 or 1) whether to use batch normalization or not
  epochs_per_run       (int) number of epochs for each fitting run
  learning_rate_start  (float) initial learning rate
  no_of_runs           (int) number of fitting runs

==================================================================================================

SAMPLE ARGUMENTS: [!NOTE: Running without batch norm takes a greater number of runs for the accuracy to increase!]
CIFAR_pa4.py 64 0 10 1e-1 5
CIFAR_pa4.py 64 1 20 1e-3 3
CIFAR_pa4.py 128 1 10 1e-2 5