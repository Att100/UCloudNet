import paddle
from paddle.io import DataLoader
import paddle.nn.functional as F
import paddle.optimizer as optim
import argparse

from models.ucloudnet import UCloudNet, _bce_loss_with_aux
from utils.dataset import SWINySEG
from utils.progressbar import ProgressBar


def bce_loss(pred, target):
    return F.binary_cross_entropy(
        F.sigmoid(paddle.squeeze(pred, 1)), 
        target)

def bce_loss_with_aux(pred, target, weight=[1, 0.4, 0.2]):
    return _bce_loss_with_aux(pred, paddle.unsqueeze(target, 1), weight)

def accuracy(pred, label, aux):
    if aux == 0:
        pred_t = F.sigmoid(paddle.squeeze(pred, 1))
    else:
        pred_t = F.sigmoid(paddle.squeeze(pred[0], 1))
    return float(
        paddle.mean(((pred_t>0.5).astype('int64')==label).astype('float32')))

def train(args):
    print("# =============== Training Configuration =============== #")
    print("# model tag: "+args.model_tag)
    print("# k: "+str(args.k))
    print("# learning rate: "+str(args.lr))
    print("# learning rate decay: "+str(bool(args.lr_decay)))
    print("# epochs: "+str(args.epochs))
    print("# dataset: SWINySEG ("+args.dataset_split+")")
    print("# aux: "+str(bool(args.aux)))
    print("# evaluation interval: "+str(args.eval_interval))
    print("# ====================================================== #")

    paddle.seed(999)

    train_set = SWINySEG(args.dataset_path, args.dataset_split, 'train')
    test_set = SWINySEG(args.dataset_path, args.dataset_split, 'test')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    train_record_file = open('./results/{}_train_log.csv'.format(args.model_tag), 'w')
    eval_record_file = open('./results/{}_eval_log.csv'.format(args.model_tag), 'w')
    
    eval_record_file.write("epoch,iteration,loss,acc\n")

    model = UCloudNet(3, args.k, 2)

    if args.aux == 1:
        train_record_file.write("epoch,iteration,loss,x1_loss,x2_loss,x4_loss,acc\n")
    else:
        train_record_file.write("epoch,iteration,loss,acc\n")

    if args.lr_decay == 1:
        scheduler = paddle.optimizer.lr.ExponentialDecay(learning_rate=args.lr, gamma=0.95, verbose=True)
        optimizer = optim.Adam(scheduler, parameters=model.parameters(), weight_decay=1e-3)
    else:
        optimizer = optim.Adam(args.lr, parameters=model.parameters(), weight_decay=1e-3)

    train_steps = len(train_loader)
    test_steps = len(test_loader)

    for e in range(args.epochs):
        train_loss = 0
        test_loss = 0
        train_acc = 0
        test_acc = 0

        bar = ProgressBar(maxStep=train_steps)
        model.train()

        for i, (image, label) in enumerate(train_loader()):
            optimizer.clear_grad()
            pred = model(image)

            if args.aux == 0:
                loss = bce_loss(pred, label)
            else:
                loss, (_1x_loss, _2x_loss, _4x_loss) = bce_loss_with_aux(pred, label)

            loss.backward()
            optimizer.step()

            batch_loss = loss.numpy()[0]
            batch_acc = accuracy(pred, label, args.aux)
            train_loss += batch_loss
            train_acc += batch_acc

            if i != train_steps-1:
                bar.updateBar(
                        i+1, headData={'Epoch':e+1, 'Status':'training'}, 
                        endData={
                            'Train loss': "{:.5f}".format(train_loss/(i+1)),
                            'Train Acc': "{:.5f}".format(train_acc/(i+1))})
            else:
                bar.updateBar(
                        i+1, headData={'Epoch':e+1, 'Status':'finished'}, 
                        endData={
                            'Train loss': "{:.5f}".format(train_loss/(i+1)),
                            'Train Acc': "{:.5f}".format(train_acc/(i+1))})
            
            if args.aux == 0:
                train_record_file.write(
                    "{},{},{},{}\n".format(e+1, e*train_steps+i+1, batch_loss, batch_acc))
            else:
                train_record_file.write(
                    "{},{},{},{},{},{},{}\n".format(
                        e+1, e*train_steps+i+1, 
                        batch_loss, 
                        _1x_loss.numpy()[0], 
                        _2x_loss.numpy()[0], 
                        _4x_loss.numpy()[0], 
                        batch_acc))

        if (e+1) % args.eval_interval == 0:
            bar = ProgressBar(maxStep=test_steps)
            model.eval()

            for i, (image, label) in enumerate(test_loader()):
                pred = model(image)

                if args.aux == 0:
                    loss = bce_loss(pred, label)
                else:
                    loss, (_1x_loss, _2x_loss, _4x_loss) = bce_loss_with_aux(pred, label)

                test_loss += loss.numpy()[0]
                test_acc += accuracy(pred, label, args.aux)

                if i != test_steps-1:
                    bar.updateBar(
                            i+1, headData={'Epoch (Test)':e+1, 'Status':'testing'}, 
                            endData={
                                'Test loss': "{:.5f}".format(test_loss/(i+1)),
                                'Test Acc': "{:.5f}".format(test_acc/(i+1))})
                else:
                    bar.updateBar(
                            i+1, headData={'Epoch (Test)':e+1, 'Status':'finished'}, 
                            endData={
                                'Test loss': "{:.5f}".format(test_loss/(i+1)),
                                'Test Acc': "{:.5f}".format(test_acc/(i+1))})

            eval_record_file.write(
                "{},{},{},{}\n".format(e+1, (e+1)*train_steps, test_loss/test_steps, test_acc/test_steps))
        
        if args.lr_decay == 1:
            scheduler.step()

    paddle.save(
        model.state_dict(), 
        "./weights/{}_epochs_{}.pdparam".format(args.model_tag, args.epochs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_tag', type=str, default='ucloudnet_k_2_aux_lr_decay', 
        help="the tag of model (default: ucloudnet_k_2_aux_lr_decay)")
    parser.add_argument(
        '--k', type=int, default=2, 
        help="the k value of model (default: 2)")
    parser.add_argument(
        '--batch_size', type=int, default=16, 
        help="batchsize for model training (default: 16)")
    parser.add_argument(
        '--lr', type=float, default=1e-3, 
        help="the learning rate for training (default: 1e-3)")
    parser.add_argument(
        '--lr_decay', type=int, default=1, 
        help="enable learning rate decay when training, [1, 0] (default: 1)")
    parser.add_argument(
        '--aux', type=int, default=1, 
        help="enable deep supervision when training, [1, 0] (default: 1)")
    parser.add_argument(
        '--epochs', type=int, default=100, 
        help="number of training epochs (default: 100)")
    parser.add_argument(
        '--dataset_split', type=str, default='all',
        help="split of SWINySEG dataset, ['all', 'd', 'n'] (default: all)")
    parser.add_argument(
        '--dataset_path', type=str, default='./dataset/SWINySEG', 
        help="path of training dataset (default: ./dataset/SWINySEG)")
    parser.add_argument(
        '--eval_interval', type=int, default=5, 
        help="interval of model evaluation during training (default: 5)"
    )
    
    args = parser.parse_args()
    
    train(args)