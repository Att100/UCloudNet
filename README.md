# UCloudNet: A residual U-Net with deep supervision for cloud segmentation

### Executive summary
In recent years, there is a growing tendency among the research of ground-based cloud image segmentation in meteorology area. A great number of researches based on traditional computer vision methods are released, which only consider simple feature of images, for example, color features and gradient variation of image after gray-scale preprocessing. With the development of deep learning in computer vision area, the CNN-based approaches are more likely to gain better performance on cloud segmentation. However, recent-years research that involve CNNs show that training consumption can be a limitation which always need thousands epochs to converge. In this paper, we introduce a residual U-Net with deep supervision for cloud segmentation which is proved to have better performance than other CNN-based approaches with less training consumption.

### Code
* `./models/`: This folder contains UCloudNet model code.
* `./utils/`: 
* `./weights/`: This folder contains the weights after model training.
* `notebook.ipynb`:
* `train.py`:


### Environment and Preparation

### Data
* `./dataset/`: This folder contains day-time images (augmented SWIMSEG), night-time images (augmented SWINSEG), and full SWINySEG dataset.

### Model
* `UCloudNet Architecture.png`: It shows the architecture overview of proposed UCloudNet. Our UCloudNet is based on the U-Net structure which contains a series of decoders and encoders with channels concatenation in each stage. To compare with the original U-Net structure, we use a hyper-parameter $k$ to control the parameters amount and inspired by K. He et al., we add residual connection in each convolution block in encoder which is helpful for training the deeper layers. As for the training strategy, we use deep supervision to support the training process.

<div align=center><img src="https://github.com/Att100/UCloudNet/blob/main/UCloudNet%20Architecture.png?raw=true" width="700"/></div>

### Training

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

### Testing

```
# follow instructions in notebook.ipynb
```

### Results
* `Results of cloud segmentation.png`: This figure shows the results of cloud segmentation for day-time (1-6 columns) and night-time (7-12 columns).
<div align=center><img src="https://github.com/Att100/UCloudNet/blob/main/Results%20of%20cloud%20segmentation.png?raw=true" width="900"/></div>

* `PR curve of UCloudNet.png`: This figure shows the PR curve of our proposed model with different training configuration on full SWINySEG ground-based cloud seg-mentation data set.

* `Loss curve of the final output and auxiliary outputs.png`: This figure shows the training status of our proposed model qualitatively by observe curves of final loss together with auxiliary loss branches
