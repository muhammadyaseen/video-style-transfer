import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import argparse

MODELS_DIR = "./models"
IMG_SIZE = 256



def parse_cmd_line_args_and_run():

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="Run the script in train mode", action="store_true")
    parser.add_argument("--transfer", help="Run the script in style transfer mode", action="store_true")
    parser.add_argument("--style", help="Absolute path to style image. Required only at train time")
    parser.add_argument("--content", help="Absolute path to content image. "
                                "This image will be transformed to match style."
                                "Required only at style time")

    args = parser.parse_args()

    if args.train and len(args.style) != 0 :
        # run in train mode
        pass

    if args.style and len(args.content) != 0 :
        # run in style mode
        pass


def train():
    # Get the pretrained VGG model. It will be used as "loss network" as in
    # perceptual loss paper
    VGGNet = torchvision.models.vgg19(pretrained=True)

def transfer(content_img_path):
    pass

def save_model(models_dir=MODELS_DIR):
    pass

def load_model(models_dir=MODELS_DIR):
    pass



class GramMatrix(nn.Module):

    def forward(self, input):

        a,b,c,d = input.size()
        features = input.view(a*b*c*d)
        G = torch.mm(features, features.t())
        return G.div(a*b*c*d)

class TransformNet(object):

    def __init__(self, style):

        super(TransformNet, self).__init__()
        self.style = style
        #self.content = content
        #self.pastiche = nn.Parameter(pastiche.data)

        self.content_layers = ['conv_4']
        self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        self.content_weight = 1
        self.style_weight = 1000

        self.loss_network = models.vgg19(pretrained=True)

        self.transform_network = nn.Sequential(nn.ReflectionPad2d(40),
                                    nn.Conv2d(3, 32, 9, stride=1, padding=4),
                                    nn.Conv2d(32, 64, 3, stride=2, padding=1),
                                    nn.Conv2d(64, 128, 3, stride=2, padding=1),
                                    nn.Conv2d(128, 128, 3, stride=1, padding=0),
                                    nn.Conv2d(128, 128, 3, stride=1, padding=0),
                                    nn.Conv2d(128, 128, 3, stride=1, padding=0),
                                    nn.Conv2d(128, 128, 3, stride=1, padding=0),
                                    nn.Conv2d(128, 128, 3, stride=1, padding=0),
                                    nn.Conv2d(128, 128, 3, stride=1, padding=0),
                                    nn.Conv2d(128, 128, 3, stride=1, padding=0),
                                    nn.Conv2d(128, 128, 3, stride=1, padding=0),
                                    nn.Conv2d(128, 128, 3, stride=1, padding=0),
                                    nn.Conv2d(128, 128, 3, stride=1, padding=0),
                                    nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                                    nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
                                    nn.Conv2d(32, 3, 9, stride=1, padding=4),
                                )

        self.gram = GramMatrix()
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.transform_network.parameters(), lr=1e-3)


    def train(self, content):

            self.optimizer.zero_grad()
            content = content.clone()
            style = self.style.clone()
            pastiche = self.transformation_network.forward(content)

            content_loss = 0
            style_loss = 0

            i = 1
            not_inplace = lambda layer: nn.ReLU(inplace=False) if isinstance(layer, nn.ReLU) else layer
            for layer in list(self.loss_network.features):
                layer = not_inplace(layer)

                pastiche, content, style = layer.forward(pastiche), layer.forward(content), layer.forward(style)

                if isinstance(layer, nn.Conv2d):
                    name = "conv_" + str(i)
                    if name in self.content_layers:
                        content_loss += self.loss(pastiche * self.content_weight, content.detach() * self.content_weight)
                    if name in self.style_layers:
                        pastiche_g, style_g = self.gram.forward(pastiche), self.gram.forward(style)
                        style_loss += self.loss(pastiche_g * self.style_weight, style_g.detach() * self.style_weight)

                if isinstance(layer, nn.ReLU):
                    i += 1

            total_loss = content_loss + style_loss

            total_loss.backward()
            self.optimizer.step()

            return pastiche
