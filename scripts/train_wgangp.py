import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from qcs.fanogan.model import Generator, Discriminator, initialize_weights
from qcs.fanogan.utils import gradient_penalty, save_checkpoint


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--image-folder', required=True, type=str, help='image folder')
    parser.add_argument('--output-folder', required=True, type=str, help='where to store the output')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--batch-size', default=64, type=int, help='batch size')
    parser.add_argument('--image-size-for-gan', default=64, type=int, help='image size for GAN only')
    parser.add_argument('--latent-dim', default=128, type=int, help='latent space dimensions')
    parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--features-critic', default=16, type=int, help='feature maps for critic')
    parser.add_argument('--features-gen', default=16, type=int, help='feature maps for generator')
    parser.add_argument('--critic-iter', default=5, type=int, help='number of critic optim iter befor optimizing gen')
    parser.add_argument('--lambda-gp', default=10, type=float, help='weight for the GP term')
    args = parser.parse_args()

    IMAGE_FOLDER = args.image_folder
    assert os.path.exists(IMAGE_FOLDER), 'image folder doesn\'t exists!'

    OUTPUT_FOLDER = args.output_folder
    assert os.path.exists(OUTPUT_FOLDER), 'output folder doesn\'t exists!'


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    LEARNING_RATE       = args.lr
    BATCH_SIZE          = args.batch_size
    IMAGE_SIZE          = args.image_size_for_gan
    CHANNELS_IMG        = 1 # Fixed for 2D MRI slices.
    Z_DIM               = args.latent_dim
    NUM_EPOCHS          = args.epochs
    FEATURES_CRITIC     = args.features_critic
    FEATURES_GEN        = args.features_gen 
    CRITIC_ITERATIONS   = args.critic_iter
    LAMBDA_GP           = args.lambda_gp


    transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
            ),
        ]
    )

    dataset = ImageFolder(root=IMAGE_FOLDER, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
    critic = Discriminator(CHANNELS_IMG, FEATURES_CRITIC).to(device)
    initialize_weights(gen)
    initialize_weights(critic)

    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
    opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))


    logdir_path = os.path.join(OUTPUT_FOLDER, 'logs', 'WGANGP')
    if not os.path.exists(logdir_path): os.makedirs(logdir_path)

    models_path = os.path.join(OUTPUT_FOLDER, 'models', 'WGANGP')
    if not os.path.exists(models_path): os.makedirs(models_path)

    fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)
    writer_real = SummaryWriter(os.path.join(logdir_path, 'real'))
    writer_fake = SummaryWriter(os.path.join(logdir_path, 'fake'))
    step = 0


    gen.train()
    critic.train()

    for epoch in range(NUM_EPOCHS):
        for batch_idx, (real, _) in enumerate(tqdm(loader)):
            real = real.to(device)
            cur_batch_size = real.shape[0]

            # training critic for CRITIC_ITERATIONS
            for _ in range(CRITIC_ITERATIONS):
                noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
                fake = gen(noise)
                
                critic_real = critic(real).reshape(-1)
                critic_fake = critic(fake).reshape(-1)
                gp = gradient_penalty(critic, real, fake, device=device)
                loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp

                critic.zero_grad()
                loss_critic.backward(retain_graph=True)
                opt_critic.step()

            gen_fake = critic(fake).reshape(-1)
            loss_gen = -torch.mean(gen_fake)
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()
                

        print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
            )
    
        with torch.no_grad():
            fake = gen(fixed_noise)
            # take out (up to) 32 examples
            img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
            img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
            writer_real.add_image("Real", img_grid_real, global_step=step)
            writer_fake.add_image("Fake", img_grid_fake, global_step=step)

        step += 1

        if (epoch + 1) % 25 == 0 or (epoch + 1) == NUM_EPOCHS:
            save_checkpoint(gen.state_dict(), os.path.join(models_path, f'gen_epoch_{epoch+1}.ckpt'))
            save_checkpoint(critic.state_dict(), os.path.join(models_path, f'critic_epoch_{epoch+1}.ckpt'))