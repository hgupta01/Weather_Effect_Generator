import torch
import torch.nn as nn
from PIL import Image
from tqdm.auto import tqdm
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms


def pil2tensor(pil:Image)->torch.Tensor:
    return transforms.functional.to_tensor(pil)


def tensor2pil(tensor:torch.Tensor)->Image:
    return transforms.functional.to_pil_image(tensor)


def load_style_transfer_model(pretrained:str=None)->nn.Module:
    if pretrained:
        print(f"Loading VGG with {pretrained} weights.")
        cnn = models.vgg19(weights=None).features
        state_dict = torch.load(pretrained)
        state_dict = {k.replace('features.', ''):v for k, v in state_dict.items() if 'features' in k}
        cnn.load_state_dict(state_dict)
    else:
        print(f"Loading VGG with IMAGENET1K weights.")
        cnn = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
    cnn.eval()
    return cnn


def style_content_image_loader(content_path, style_path):
    wreq = 640

    content_img = Image.open(content_path)
    wc, hc = content_img.size
    wc_new, hc_new = wreq, int(hc*wreq/wc)
    content_img = content_img.resize((wc_new, hc_new))
    
    
    style_img = Image.open(style_path)
    ws, hs = style_img.size
    
    ws_new = wreq
    hs_new = int(hs*ws_new/ws)

    if hs_new < hc_new:
        hs_new = hc_new

    style_img = style_img.resize((ws_new, hs_new))

    if hs_new>hc_new:
        top = int((hs_new - hc_new )*0.5)
        bottom = top+hc_new
        style_img = style_img.crop((0, top, ws_new, bottom))

    assert style_img.size == content_img.size
    
    style_img = pil2tensor(style_img).unsqueeze(0)
    content_img = pil2tensor(content_img).unsqueeze(0)
    return content_img, style_img


class ContentLoss(nn.Module):
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        # self.target = target.detach()
        self.register_buffer('target', target.detach())

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        #self.target = gram_matrix(target_feature).detach()
        self.register_buffer('target', gram_matrix(target_feature).detach())

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

    
def get_style_model_and_losses(cnn, style_img, content_img, device="cpu"):
    # desired depth layers to compute style/content losses :
    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    # just in order to have an iterable access to or list of content/syle losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    model = nn.Sequential(transforms.Normalize(mean=mean, std=std))

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]
    for sl in style_losses:
        sl.to(device)
    for sl in content_losses:
        sl.to(device)
    return model.to(device), style_losses, content_losses


def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = torch.optim.LBFGS([input_img]) #, lr=1e-2
    return optimizer


def run_style_transfer(cnn, content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1, device="cpu"):
    """Run the style transfer."""
    # print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn, style_img, content_img, device=device)

    # We want to optimize the input and not the model parameters so we
    # update all the requires_grad fields accordingly
    input_img.requires_grad_(True)
    model.requires_grad_(False)

    optimizer = get_input_optimizer(input_img)

    for run in tqdm(range(num_steps)):
        def closure():
            # correct the values of updated input image
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()
            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img