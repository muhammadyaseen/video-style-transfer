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

# Get paths and set vars
#weights_fname = "candy.pth"
tsfm_weights = "third.model"
mnet_weights = "mnet2.model"

script_path = os.path.dirname(os.path.abspath(__file__))
path_to_tfsm_weights = os.path.join(script_path, "model", tsfm_weights)
path_to_mnet_weights = os.path.join(script_path, "model", mnet_weights)
resolution = (600, 480)

# Change to GPU if desired
#device = torch.device("cuda")
device = torch.device("cuda")

# Load PyTorch Model
tfsm = TransformerNet()
mnet = DSC_TransformerNet()

with torch.no_grad():
   state_dict = torch.load(path_to_tfsm_weights)
   for k in list(state_dict.keys()):
        if re.search(r'in\d+\.running_(mean|var)$', k):
            del state_dict[k]
   tfsm.load_state_dict(state_dict)
   tfsm.to(device)

with torch.no_grad():
   state_dict = torch.load(path_to_mnet_weights)
   for k in list(state_dict.keys()):
        if re.search(r'in\d+\.running_(mean|var)$', k):
            del state_dict[k]
   mnet.load_state_dict(state_dict)
   mnet.to(device)

# Get Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("OpenCV cannot find your webcam! Check that it is under /dev/video0")
    exit(1)

while True:
    # Grab frame and change to jpeg
    ret, frame = cap.read()
    cv2_im = frame #cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im)
    img = pil_im.resize(resolution)

    # Transforms to feed to network
    small_frame_tensor_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    small_frame_tensor = small_frame_tensor_transform(img)
    small_frame_tensor = small_frame_tensor.unsqueeze(0).to(device)

    # Run inference and resize
    output_tfsm = tfsm(small_frame_tensor).cpu()
    output_mnet = mnet(small_frame_tensor).cpu()

    styled_tfsm = output_tfsm[0]
    styled_tfsm = styled_tfsm.clone().clamp(0, 255).detach().numpy()
    styled_tfsm = styled_tfsm.transpose(1, 2, 0).astype("uint8")
    styled_resized_tfsm = cv2.resize(cv2.cvtColor(styled_tfsm, cv2.COLOR_RGB2BGR),(frame.shape[0], frame.shape[1]))

    styled_mnet = output_mnet[0]
    styled_mnet = styled_mnet.clone().clamp(0, 255).detach().numpy()
    styled_mnet = styled_mnet.transpose(1, 2, 0).astype("uint8")
    styled_resized_mnet = cv2.resize(cv2.cvtColor(styled_mnet, cv2.COLOR_RGB2BGR),(frame.shape[0], frame.shape[1]))

    del styled_tfsm
    del output_tfsm
    del styled_mnet
    del output_mnet

    # Display frame and break if user hits q
    cv2.imshow('Transform net', styled_resized_tfsm)
    cv2.imshow('MNet', styled_resized_mnet)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
