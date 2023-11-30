import os
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from qcs.fanogan.model import Encoder, Generator, Discriminator
from qcs.fanogan.utils import save_checkpoint

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--image-folder', required=True, type=str, help='image folder')
    parser.add_argument('--output-folder', required=True, type=str, help='where to store the output')
    parser.add_argument('--gen', required=True, type=str, help='generator checkpoints path')
    parser.add_argument('--critic', required=True, type=str, help='critic checkpoints path')
    parser.add_argument('--lr', default=2e-4, type=float, help='learning rate')
    parser.add_argument('--k', default=1., type=float, help='k hyperparameter')
    parser.add_argument('--batch-size', default=64, type=int, help='batch size')
    parser.add_argument('--image-size-for-gan', default=64, type=int, help='image size for gan')
    parser.add_argument('--latent-dim', default=128, type=int, help='latent space dimensions')
    parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--features-critic', default=16, type=int, help='feature maps for critic')
    parser.add_argument('--features-gen', default=16, type=int, help='feature maps for generator')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    LEARNING_RATE   = args.lr
    Z_DIM           = args.latent_dim
    CHANNELS_IMG    = 1
    FEATURES_GEN    = args.features_gen
    FEATURES_CRITIC = args.features_critic
    IMAGE_SIZE      = args.image_size_for_gan
    NUM_EPOCHS      = args.epochs
    BATCH_SIZE      = args.batch_size
    K               = args.k

    IMAGE_FOLDER = args.image_folder
    assert os.path.exists(IMAGE_FOLDER), 'image folder doesn\'t exists!'

    OUTPUT_FOLDER = args.output_folder
    assert os.path.exists(OUTPUT_FOLDER), 'output folder doesn\'t exists!'

    GEN_CKPT_PATH = args.gen
    assert os.path.exists(GEN_CKPT_PATH), 'invalid generator checkpoints path'

    CRITIC_CKPT_PATH = args.critic
    assert os.path.exists(CRITIC_CKPT_PATH), 'invalid critic checkpoints path'

    enc     = Encoder(CHANNELS_IMG, IMAGE_SIZE, Z_DIM)
    gen     = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
    critic  = Discriminator(CHANNELS_IMG, FEATURES_CRITIC).to(device)

    # Loading Gen and Critic
    gen.load_state_dict(torch.load(GEN_CKPT_PATH))
    critic.load_state_dict(torch.load(CRITIC_CKPT_PATH))
    
    enc = enc.to(device)
    gen = gen.to(device).eval()
    critic = critic.to(device).eval()

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([.5], [.5])
    ])

    dataset = ImageFolder(root=IMAGE_FOLDER, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    logdir_path = os.path.join(OUTPUT_FOLDER, 'logs', 'IZIF')
    if not os.path.exists(logdir_path): os.makedirs(logdir_path)

    models_path = os.path.join(OUTPUT_FOLDER, 'models', 'IZIF')
    if not os.path.exists(models_path): os.makedirs(models_path)

    writer_real = SummaryWriter(os.path.join(logdir_path, 'real'))
    writer_fake = SummaryWriter(os.path.join(logdir_path, 'fake'))

    # getting 16 real images
    real_images = []
    for i in range(16): real_images.append(dataset[i][0]) 
    fixed_real_images = torch.stack(real_images, dim=0).to(device)

    loss_fun = nn.MSELoss()
    optmizer = torch.optim.Adam(enc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    step = 0
    loss_avg = []

    for epoch in range(NUM_EPOCHS):
        for i, (real, _) in enumerate(loader):
            real = real.to(device)

            optmizer.zero_grad()
            z = enc(real)
            fake = gen(z.view(-1, Z_DIM, 1, 1))

            real_features = critic.forward_features(real)
            fake_features = critic.forward_features(fake)

            loss_images = loss_fun(fake, real)
            loss_features = loss_fun(fake_features, real_features)
            
            loss = loss_images + K * loss_features
            loss.backward()
            optmizer.step()

            loss_avg.append(loss.item())
    
        print('epoch', epoch, 'avg loss', sum(loss_avg) / len(loss_avg))
        with torch.no_grad():
                real = fixed_real_images
                z = enc(real)
                fake = gen(z.view(-1, Z_DIM, 1, 1))
                img_grid_real = torchvision.utils.make_grid(real, normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)
                step += 1

        if (epoch + 1) % 25 == 0 or (epoch + 1) == NUM_EPOCHS:
            save_checkpoint(enc.state_dict(), os.path.join(models_path, f'enc_epoch_{epoch+1}.ckpt'))