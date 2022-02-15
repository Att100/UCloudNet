import paddle
import paddle.nn.functional as F
from PIL import Image
import numpy as np
import argparse

from models.unet import UNet
from models.ucloudnet import UCloudNet

def test(args):
    if args.model == 'unet':
        model = UNet(3, args.k, 2)
    elif args.model == 'ucloudnet':
        model = UCloudNet(3, args.k, 2)
    else:
        raise Exception("model name not support, ['ucloudnet', 'unet']")

    state_dict = paddle.load(
        "./weights/{}_epochs_{}.pdparam".format(args.model_tag, args.epochs))
    model.set_state_dict(state_dict)

    img = Image.open("./dataset/SWINySEG/images/d0002_1.jpg").resize((304, 304))
    img_tensor = paddle.to_tensor(np.array(img).transpose(2, 0, 1).reshape((1, 3, 304, 304))).astype('float32') / 255

    model.eval()

    pred = model(img_tensor)
    pred = (F.sigmoid(pred)>0.5).astype('int64')

    mask = Image.fromarray(np.uint8(pred.numpy().reshape((304, 304))) * 255)

    img.save("./results/test.jpg")
    mask.save("./results/test_pred_{}.jpg".format(args.model_tag))
    
    mask.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', type=str, default='ucloudnet', 
        help="the name of the model ['ucloudnet', 'unet'] (default: ucloudnet)")
    parser.add_argument(
        '--model_tag', type=str, default='ucloudnet_k_2_aux_lr_decay', 
        help="the tag of model (default: ucloudnet_k_2_aux_lr_decay)")
    parser.add_argument(
        '--k', type=int, default=2, 
        help="the k value of model (default: 2)")
    parser.add_argument(
        '--epochs', type=int, default=100, 
        help="number of training epochs (default: 100)")
    
    args = parser.parse_args()

    test(args)

    
