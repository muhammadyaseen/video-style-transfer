#!/usr/bin/env python3
import os
import re
import numpy as np
from PIL import Image
import cv2
import torch
from torchvision import transforms
from model.transformer_net import TransformerNet
from model.dsc_transformer import DSC_TransformerNet
import matplotlib.pyplot as plt

import argparse

arg_parser = argparse.ArgumentParser(description="parser for fast-neural-style")
arg_parser.add_argument('--arch', choices=['tf', 'mnet'])
arg_parser.add_argument('--model', required=True, help="full path to model file")
args = arg_parser.parse_args()

# Get paths and set vars
#weights_fname = "candy.pth"
weights_fname = args.model
path_to_weights = weights_fname

#weights_fname = "third.model" if args.model == "tf" else "mnet2.model"
#weights_fname = "mnet2.model"
#script_path = os.path.dirname(os.path.abspath(__file__))
#path_to_weights = os.path.join(script_path, "model", weights_fname)
resolution = (800, 600)

# Change to GPU if desired
#device = torch.device("cuda")
device = torch.device("cuda")

# Load PyTorch Model
model = TransformerNet() if args.arch == "tf" else DSC_TransformerNet()
#model = DSC_TransformerNet()
with torch.no_grad():
   state_dict = torch.load(path_to_weights)
   for k in list(state_dict.keys()):
        if re.search(r'in\d+\.running_(mean|var)$', k):
            del state_dict[k]
   model.load_state_dict(state_dict)
   model.to(device)

# Get Webcam
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('v2.mp4')

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (800,600))

if not cap.isOpened():
    print("OpenCV cannot find your webcam! Check that it is under /dev/video0")
    exit(1)

cv2.startWindowThread()
cv2.namedWindow(args.model)

kernel = np.ones((10,10),np.uint8)
i=0
while True:
    # Grab frame and change to jpeg

    ret, frame = cap.read()
    cv2_img = frame #cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    #blur = cv2.GaussianBlur(cv2_img, (31,31), 0)
    #cv2_img = cv2.morphologyEx(cv2_img, cv2.MORPH_CLOSE, kernel)
    cv2_img = cv2.GaussianBlur(cv2_img, (17,17), 10)

    #cv2_img = cv2.morphologyEx(cv2_img, cv2.MORPH_CLOSE, kernel)

    pil_im = Image.fromarray(cv2_img)
    img = pil_im.resize(resolution)

    # Transforms to feed to network
    small_frame_tensor_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    small_frame_tensor = small_frame_tensor_transform(img)
    small_frame_tensor = small_frame_tensor.unsqueeze(0).to(device)

    # Run inference and resize
    output = model(small_frame_tensor).cpu()
    styled = output[0]
    styled = styled.clone().clamp(0, 255).detach().numpy()
    styled = styled.transpose(1, 2, 0).astype("uint8")
    #styled_resized = cv2.resize(cv2.cvtColor(styled, cv2.COLOR_RGB2BGR),(frame.shape[0], frame.shape[1]))
    styled_resized = cv2.resize(cv2.cvtColor(styled, cv2.COLOR_RGB2BGR), resolution)
    out.write(styled_resized)
    #styled_resized = cv2.morphologyEx(styled_resized, cv2.MORPH_CLOSE, kernel)
    #styled_resized = cv2.GaussianBlur(styled_resized, (31,31), 0)

    del styled
    del output

    # Display frame and break if user hits q
    #cv2.imshow(args.model, cv2_img)
    #cv2.imshow(args.arch, styled_resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
out.release()
