#!/bin/bash

run_mode="train"
mode_set=false

epochs=10
t_size=16
v_size=32
config_file=/home/angelman7/PycharmProjects/D-FINE/configs/dfine/custom/dfine_hgnetv2_n_custom.yml

while [[ $# -gt 0 ]]; do
  case "$1" in
    -train)
      if $mode_set; then
        echo "Error: cannot use both -train and -test"
        exit 1
      fi
      run_mode="train"
      mode_set=true
      shift
      ;;
    -test)
      if $mode_set; then
        echo "Error: cannot use both -train and -test"
        exit 1
      fi
      run_mode="test"
      mode_set=true
      shift
      ;;
    -e|--epochs)
      epochs="$2"
      shift 2
      ;;
    -tb|--train-batch-size)
      t_size="$2"
      shift 2
      ;;
    -vb|--validation-batch-size)
      v_size="$2"
      shift 2
      ;;
    -conf|--config-file)
      config_file="$2"
      shift 2
      ;;
    *)
      echo ""
      echo "-------------------------------------------------Supported flags--------------------------------------------------"
      echo "  Mode:"
      echo "      -train : Trains the model using the training data from the dataset provided"
      echo "      -test  : Tests the model using the validation data from the dataset provided"
      echo "  Options:"
      echo "      -e | --epochs  <epochs>                               : Select the number of epochs (Default=10)"
      echo "      -lr | --learing-rate  <learing-rate>                  : Choose the learing rate of the model (Default=0.0008)"
      echo "      -tb | --train_batch_size <train_batch_size>           : Select the size of the train batch (Default=16)"
      echo "      -vb | --validation_batch_size <validation_batch_size> : Select the size of the validation batch (Default=32)"
      echo "      -conf | --config-file <filename>                      : Select the config file to use (Default=D-FINE custom"
      echo ""
      echo "------------------------------------------------------------------------------------------------------------------"
      exit 1
      ;;
  esac
done


case "$run_mode" in
  train|test) ;;
  *)
    echo "Invalid option. Choose: train | test"
    exit 1
    ;;
esac

sed -i "70s/.*/epochs: $epochs/" $config_file
sed -i "72s/.*/  total_batch_size: $t_size/" $config_file
sed -i "83s/.*/  total_batch_size: $v_size/" $config_file

source /home/angelman7/miniconda3/etc/profile.d/conda.sh
conda activate dfine

wandb login

case "$run_mode" in
  train)
    CUDA_VISIBLE_DEVICES=0 torchrun --master_port=7777 --nproc_per_node=1 train.py -c configs/dfine/custom/dfine_hgnetv2_n_custom.yml --use-amp --seed=0
    ;;
  test)
    CUDA_VISIBLE_DEVICES=0 torchrun --master_port=7777 --nproc_per_node=1 train.py -c configs/dfine/custom/dfine_hgnetv2_n_custom.yml --test-only -r output/cone_n_custom/best_stg1.pth
    ;;
esac

conda deactivate

set +e
$@
notify-send "Training Completed!"
