import os
import argparse
import pickle
from multiprocessing import Pool

import scipy
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as TVF
from torchvision.datasets import ImageFolder

from qcs.fanogan.model import Discriminator
from qcs.artefacts.factory import load_class

# ---------------------------------------------------
# GLOBAL HYPERPARAMETERS
# ---------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--image-folder', type=str, required=True, help='Image folder')
parser.add_argument('--output-folder', required=True, type=str, help='where to store the output')
parser.add_argument('--artefact', required=True, type=str, help='name of the artefact')
parser.add_argument('--critic', type=str, required=True, help='path for checkpoints of critic')
parser.add_argument('--features-critic', type=int, default=16, help='critic feature maps')
parser.add_argument('--batch-size', default=64, type=int, help='batch size')
parser.add_argument('--image-size-for-gan', default=64, type=int, help='image size for gan')
parser.add_argument('--procs', type=int, default=10, help='number of processes / starting points')
args = parser.parse_args()


IMAGE_FOLDER = args.image_folder
assert os.path.exists(IMAGE_FOLDER), 'image folder doesn\'t exists!'

OUTPUT_FOLDER = args.output_folder
assert os.path.exists(OUTPUT_FOLDER), 'output folder doesn\'t exists!'

CRITIC_CKPT_PATH = args.critic
assert os.path.exists(CRITIC_CKPT_PATH), 'invalid critic checkpoints path'

CHANNELS_IMG    = 1
FEATURES_CRITIC = args.features_critic
IMAGE_SIZE      = args.image_size_for_gan
BATCH_SIZE      = args.batch_size
N_PROCS         = args.procs
ARTEFACT_NAME   = args.artefact

artefact_cls = load_class(ARTEFACT_NAME)
BOUNDS          = artefact_cls.params_bounds

if load_class(ARTEFACT_NAME).get_n_params() < 1:
    print('this artefact has not parameters to optimize.')
    exit()

# ---------------------------------------------------
# ---------------------------------------------------
# ---------------------------------------------------

def report(images_loss, params_loss, params):
    print(f'Images Loss = {images_loss:.4f} Params Loss = {params_loss:.4f}')
    print('params:', [ int(p) for p in params ])
    print('-' * 100)

def init_params(bounds):
    return np.array([ np.random.uniform(b[0], b[1]) for b in bounds ])

def loss_function(params, dataset, critic):
    global IMAGE_SIZE
    artefact = load_class(ARTEFACT_NAME)
    artefact.set_params(params)
    critic_values = []
    for i in np.random.permutation(len(dataset))[:BATCH_SIZE]:
        corrupted = artefact.transform(dataset[i][0], severity=np.random.randint(1, 10))
        # turn to tensor and add channel and batch dimensions
        corrupted_tensor = torch.tensor(corrupted.astype(float)).unsqueeze(0).unsqueeze(0)
        corrupted_tensor = TVF.resize(corrupted_tensor, size=(IMAGE_SIZE, IMAGE_SIZE), antialias=True)
        corrupted_tensor = TVF.normalize(corrupted_tensor, [.5], [.5]).float()
        critic_values.append( critic(corrupted_tensor).item() )
    images_loss = np.mean(critic_values)
    params_loss = np.sum(np.abs(params))
    report(images_loss, params_loss, params)
    return images_loss + params_loss


def process(params):
    global CHANNELS_IMG, FEATURES_CRITIC, IMAGE_FOLDER

    critic = Discriminator(CHANNELS_IMG, FEATURES_CRITIC)
    transform = transforms.Compose([ transforms.Grayscale(), transforms.Lambda(np.array) ])
    dataset = ImageFolder(root=IMAGE_FOLDER, transform=transform)

    res = scipy.optimize.minimize(loss_function, 
                                  params, 
                                  args=(dataset, critic), 
                                  bounds=BOUNDS)
    return (res.fun, res.x)


if __name__ == '__main__':

    params_set = [ init_params(BOUNDS) for _ in range(N_PROCS) ]
    
    with Pool(N_PROCS) as pool:
        results = pool.map(process, params_set)
        results_map = {}
        
        best_params, best_loss = None, None
        for i, (loss, params) in enumerate(results):
            print(f'execution {i} - loss: {loss:.3f} - params', [round(p, 2) for p in params ])
            results_map[i] = { 'loss': loss, 'params': params }
            
            if best_loss is None or best_loss > loss:
                best_loss = loss
                best_params = params

        output_dir = os.path.join(OUTPUT_FOLDER, 'params_optim')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        pickle.dump(results_map, open(os.path.join(output_dir, f'{ARTEFACT_NAME}_runs-results.pkl'), 'wb'))
        np.save(os.path.join(output_dir, f'{ARTEFACT_NAME}_best_params.npy'), best_params)