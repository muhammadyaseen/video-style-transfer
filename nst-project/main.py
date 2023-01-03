from transfer import TransformNet
import torchvision
import torchvision.datasets as datasets
import torch.utils.data
from torch.autograd import Variable

import scipy
from PIL import Image

num_epochs = 3
N = 4
DATA_DIR = './val2017/'
IMG_SIZE = 256
loader = torchvision.transforms.Compose([
             torchvision.transforms.Scale(IMG_SIZE),
             torchvision.transforms.ToTensor()
         ])

unloader = torchvision.transforms.ToPILImage()

def image_loader(image_name):
    image = Image.open(image_name)
    image = Variable(loader(image))
    image = image.unsqueeze(0)
    return image

def save_images(input, paths):
    N = input.size()[0]
    images = input.data.clone().cpu()
    for n in range(N):
        image = images[n]
        image = image.view(3, IMG_SIZE, IMG_SIZE)
        image = unloader(image)
        scipy.misc.imsave(paths[n], image)


def main():

    style_image = image_loader("./styles/s1.jpg")
    tfsm_net = TransformNet(style)

    # Content image dataset (using MS-COCO validation set for now)
    coco_val = datasets.ImageFolder(
            root=DATA_DIR,
            transform=loader    # images are not uniform, so we need to resize
    )
    content_loader = torch.utils.data.DataLoader(coco_val, batch_size=N,
                                            shuffle=True, **kwargs)



    for epoch in range(num_epochs):
        for i, content_batch in enumerate(content_loader):
          iteration = epoch * i + i
          content_loss, style_loss, pastiches = tfsm_net.train(content_batch,
                                                                style_image)

          if i % 10 == 0:
              print("Iteration: %d" % (iteration))
              print("Content loss: %f" % (content_loss.data[0]))
              print("Style loss: %f" % (style_loss.data[0]))

          if i % 500 == 0:
              path = "outputs/%d_" % (iteration)
              paths = [path + str(n) + ".png" for n in range(N)]
              save_images(pastiches, paths)

              path = "outputs/content_%d_" % (iteration)
              paths = [path + str(n) + ".png" for n in range(N)]
              save_images(content_batch, paths)
              tfsm_net.save()

main()
