# UCloudNet: A residual U-Net with deep supervision for cloud segmentation

## 1. Introduction

## 2. Environment and Preparation

## 3. Usage

### 3.1 Training

- help

    ```
    python train.py -h

    usage: train.py [-h] [--model_tag MODEL_TAG] [--k K] [--batch_size BATCH_SIZE] [--lr LR] [--lr_decay LR_DECAY] [--aux AUX] [--epochs EPOCHS]
                    [--dataset_split DATASET_SPLIT] [--dataset_path DATASET_PATH] [--eval_interval EVAL_INTERVAL]

    optional arguments:
    -h, --help            show this help message and exit
    --model_tag MODEL_TAG
                            the tag of model (default: ucloudnet_k_2_aux_lr_decay)
    --k K                 the k value of model (default: 2)
    --batch_size BATCH_SIZE
                            batchsize for model training (default: 16)
    --lr LR               the learning rate for training (default: 1e-3)
    --lr_decay LR_DECAY   enable learning rate decay when training, [1, 0] (default: 1)
    --aux AUX             enable deep supervision when training, [1, 0] (default: 1)
    --epochs EPOCHS       number of training epochs (default: 100)
    --dataset_split DATASET_SPLIT
                            split of SWINySEG dataset, ['all', 'd', 'n'] (default: all)
    --dataset_path DATASET_PATH
                            path of training dataset (default: ./dataset/SWINySEG)
    --eval_interval EVAL_INTERVAL
                            interval of model evaluation during training (default: 5)

    ```

- experiments

    ```
    # train UCloudNet(k=2)+aux+lr_decay on full SWINySEG
    python train.py 

    # train UCloudNet(k=4)+aux+lr_decay on full SWINySEG
    python train.py --k=4 --model_tag=ucloudnet_k_4_aux_lr_decay

    # train UCloudNet(k=4)+lr_decay on full SWINySEG
    python train.py --k=4 --aux=0 --model_tag=ucloudnet_k_4_lr_decay

    # train UCloudNet(k=4) on full SWINySEG
    python train.py --k=4 --aux=0 --lr_decay=0 --model_tag=ucloudnet_k_4

    # train UCloudNet(k=2)+aux+lr_decay on SWINySEG day-time
    python train.py --k=2 --model_tag=ucloudnet_k_2_aux_lr_decay_d

    # train UCloudNet(k=2)+aux+lr_decay on SWINySEG night-time
    python train.py --k=4 --model_tag=ucloudnet_k_4_aux_lr_decay_d

    # train UCloudNet(k=4)+aux+lr_decay on SWINySEG day-time
    python train.py --k=2 --model_tag=ucloudnet_k_2_aux_lr_decay_n

    # train UCloudNet(k=4)+aux+lr_decay on SWINySEG night-time
    python train.py --k=4 --model_tag=ucloudnet_k_2_aux_lr_decay_n
    ```

### 3.2 Testing

```
# follow instructions in notebook.ipynb
```

## 4. Results