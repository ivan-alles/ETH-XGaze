import os
import glob
import argparse
import cv2
import numpy as np
import torch
from torchvision import transforms
from model import gaze_network

trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),  # this also convert pixel value from [0,255] to [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def draw_gaze(image_in, pitchyaw, thickness=2, color=(0, 0, 255)):
    """Draw gaze angle on given image with a given eye positions."""
    image_out = image_in
    (h, w) = image_in.shape[:2]
    length = w / 2.0
    pos = (int(h / 2.0), int(w / 2.0))
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[1]) * np.cos(pitchyaw[0])
    dy = -length * np.sin(pitchyaw[0])
    cv2.arrowedLine(image_out, tuple(np.round(pos).astype(np.int32)),
                   tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)), color,
                   thickness, cv2.LINE_AA, tipLength=0.2)

    return image_out


def predict(image_files, output_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    print('load gaze estimator')
    model = gaze_network()
    pre_trained_model_path = './ckpt/epoch_24_ckpt.pth.tar'
    if not os.path.isfile(pre_trained_model_path):
        print('the pre-trained gaze estimation model does not exist.')
        exit(0)
    else:
        print('load the pre-trained model: ', pre_trained_model_path)
    ckpt = torch.load(pre_trained_model_path)
    model.load_state_dict(ckpt['model_state'], strict=True)  # load the pre-trained model
    model.eval()  # change it to the evaluation mode
    model.to(device)

    for i, img_file_name in enumerate(image_files):
        image = cv2.imread(img_file_name)
        input_var = image[:, :, ::-1]  # from BGR to RGB
        input_var = trans(input_var).unsqueeze(0).to(device)
        #input_var = torch.autograd.Variable(input_var.float().cuda())
        #input_var = input_var.view(1, input_var.size(0), input_var.size(1), input_var.size(2))  # the input must be 4-dimension
        pred_gaze = model(input_var)  # get the output gaze direction, this is 2D output as pitch and raw rotation
        pred_gaze = pred_gaze[0] # here we assume there is only one face inside the image, then the first one is the prediction
        pred_gaze_np = pred_gaze.cpu().data.numpy()  # convert the pytorch tensor to numpy array

        # draw the facial landmarks
        face_patch_gaze = draw_gaze(image, pred_gaze_np)  # draw gaze direction on the normalized face image
        cv2.imwrite(os.path.join(output_path, f'{i:05d}.png'), face_patch_gaze)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default=None, help="path to image or directory with images")
    parser.add_argument("--glob_pattern", default='/**/*.png', help="pattern to search for images")
    parser.add_argument("--output_path", default='output', help="output directory")

    args = parser.parse_args()
    if os.path.isfile(args.input_path):
        image_files = [args.input_path]
    else:
        glob_pattern = args.input_path + args.glob_pattern
        image_files = glob.glob(glob_pattern, recursive=True)

    os.makedirs(args.output_path, exist_ok=True)

    predict(image_files, args.output_path)
