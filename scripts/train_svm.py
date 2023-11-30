import os
import argparse
import pickle

import numpy as np
import torch
import nibabel as nib
import skimage
from tqdm import tqdm
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from skimage.exposure import rescale_intensity

from qcs.feature_extraction.utils import extract_slices_with_dns
from qcs.artefacts.factory import load_class
from qcs.artefacts.nobrain import NoBrainTransform
from qcs.artefacts import utils
from qcs.feature_extraction import utils as fe_utils
from qcs.feature_extraction.deep import get_resnet
from qcs.feature_selection import get_feature_group
from qcs.preprocessing import percnorm, normalize_scan
from qcs.fanogan.model import Discriminator, Generator, Encoder


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--volume-folder', type=str, required=True, help='Folder containing MRI in NIfTI format')
    parser.add_argument('--output-folder', type=str, required=True, help='Output folder')
    parser.add_argument('--enc', type=str, required=True, help='Trained Encoder path')
    parser.add_argument('--gen', type=str, required=True, help='Trained Generator path')
    parser.add_argument('--critic', type=str, required=True, help='Trained Critic path')
    parser.add_argument('--features-gen', type=int, default=16, help='Generator features')
    parser.add_argument('--features-critic', type=int, default=16, help='Critic features')
    parser.add_argument('--latent-dim', type=int, default=128, help='f-AnoGAN latent dimensions')
    parser.add_argument('--artefact', type=str, required=True, help='artefact name (find table in README)')
    parser.add_argument('--artefact-params', type=str, default=None, help='path to artefact generator parameters')
    parser.add_argument('--save-synth-images', action='store_true', help='save generated artefact images')
    parser.add_argument('--severity', type=int, default=3, help='artefact severity [1-9]')
    parser.add_argument('--image-size', type=int, default=300, help='Image size')
    parser.add_argument('--image-size-for-gan', type=int, default=64, help='Image size for GAN critic')
    parser.add_argument('--slices-per-axis', type=int, default=3, help='number of slices per axis')
    parser.add_argument('--axis', type=int, default=-1, help='[0, 1, 2 or -1 for all three]')
    parser.add_argument('--nobrain-folder', type=str, help='Folder containing no-brain MRI')
    parser.add_argument('--seed', type=int, default=28, help='seed for RNG')
    parser.add_argument('--perc-norm', type=int, default=95, help='perc for percentile normalization')
    args = parser.parse_args()

    VOLUME_FOLDER = args.volume_folder
    assert os.path.exists(VOLUME_FOLDER), 'volume folder doesn\'t exists!'

    OUTPUT_FOLDER = args.output_folder
    assert os.path.exists(OUTPUT_FOLDER), 'output folder doesn\'t exists!'

    ENC_CKPT_PATH = args.enc
    assert os.path.exists(ENC_CKPT_PATH), 'invalid encoder checkpoints path'

    GEN_CKPT_PATH = args.gen
    assert os.path.exists(GEN_CKPT_PATH), 'invalid generator checkpoints path'

    CRITIC_CKPT_PATH = args.critic
    assert os.path.exists(CRITIC_CKPT_PATH), 'invalid critic checkpoints path'

    ARTEFACT_PARAMS = args.artefact_params
    assert (ARTEFACT_PARAMS is None) or os.path.exists(CRITIC_CKPT_PATH), 'invalid artefact generator parameters'

    #-------------------------------------------
    # Step 1 - Extracting the features
    #-------------------------------------------

    AXIS                = args.axis
    N_SLICES            = args.slices_per_axis
    ARTEFACT_NAME       = args.artefact
    SAVE_SYNTH_IMAGES   = args.save_synth_images & (ARTEFACT_NAME != 'nobrain') 
    SEVERITY            = args.severity
    FEATURES_CRITIC     = args.features_critic
    FEATURES_GEN        = args.features_gen
    IMAGE_SIZE          = args.image_size
    IMAGE_SIZE_GAN      = args.image_size_for_gan
    RAND_SEED           = args.seed 
    LATENT_DIM          = args.latent_dim
    PERC_NORM           = args.perc_norm
    imshape_gan         = (IMAGE_SIZE_GAN, IMAGE_SIZE_GAN)

    NOBRAIN_FOLDER      = None
    if ARTEFACT_NAME == 'nobrain':
        assert args.nobrain_folder is not None, 'need to provide no-brain images folder'
        NOBRAIN_FOLDER = args.nobrain_folder
        assert os.path.exists(NOBRAIN_FOLDER), 'invalid no-brain folder path'

    dataset_dir_path = os.path.join(OUTPUT_FOLDER, 'datasets')
    dataset_path = os.path.join(dataset_dir_path, f'{ARTEFACT_NAME}_axis_{AXIS}_nslices_{N_SLICES}_dataset.pkl')

    if not os.path.exists(dataset_path):
        
        print('Creating the dataset.')
        if not os.path.exists(dataset_dir_path): os.makedirs(dataset_dir_path)
        
        data_X, data_y = [], []

        #-------------------------------------------
        # Instantiate the artifact generator along 
        # with the optimized parameters.
        #-------------------------------------------

        if ARTEFACT_NAME != 'nobrain':

            artefact_gen = \
                load_class(ARTEFACT_NAME, SEVERITY, ARTEFACT_PARAMS) \
                if ARTEFACT_PARAMS is not None \
                else load_class(ARTEFACT_NAME, SEVERITY)

        else: artefact_gen = NoBrainTransform(NOBRAIN_FOLDER)

        #-------------------------------------------
        # Instantiate the feature selector 
        #-------------------------------------------

        feature_selector = get_feature_group(ARTEFACT_NAME)

        #-------------------------------------------
        # Instantiate the Deep Neural Networks
        #-------------------------------------------

        resnet = get_resnet()
        enc = Encoder(1, IMAGE_SIZE_GAN, LATENT_DIM)
        gen = Generator(LATENT_DIM, 1, FEATURES_GEN)
        critic = Discriminator(1, FEATURES_CRITIC)

        enc.load_state_dict(torch.load(ENC_CKPT_PATH))
        gen.load_state_dict(torch.load(GEN_CKPT_PATH))
        critic.load_state_dict(torch.load(CRITIC_CKPT_PATH))

        #-------------------------------------------
        # Begin the feature extraction
        #-------------------------------------------

        # save the generated images if specified.
        synth_store_path = os.path.join(OUTPUT_FOLDER, 'synth-images', ARTEFACT_NAME)
        if SAVE_SYNTH_IMAGES and not os.path.exists(synth_store_path):
            os.makedirs(synth_store_path)

        files = [ f for f in os.listdir(VOLUME_FOLDER) if f.endswith('.nii') or f.endswith('.nii.gz') ]

        for i, filename in enumerate(tqdm(files)):
            
            #-------------------------------------------
            # Load the MRI volume from the NIfTI file 
            # and extract the 2D slices.
            #-------------------------------------------
            
            mri_path = os.path.join(VOLUME_FOLDER, filename)
            mri = nib.load(mri_path)
            mri_data = percnorm(mri.get_fdata(), perc=PERC_NORM)

            norm_slices = []
            if AXIS in [0, 1, 2]:
                norm_slices += extract_slices_with_dns(mri_data, n_slices=N_SLICES, force_axis=AXIS)[0]
            else:
                for ax in range(3):
                    # Extract N_SLICE for every axis.
                    norm_slices += extract_slices_with_dns(mri_data, n_slices=N_SLICES, force_axis=ax)[0]

            #-------------------------------------------
            # Apply some preprocessing to the slice and 
            # generate the artefact.
            #-------------------------------------------

            norm_slices_features = []
            synt_slices_features = []

            for j, norm_s in enumerate(norm_slices):

                norm_s = rescale_intensity(normalize_scan(norm_s, img_size=IMAGE_SIZE), out_range=np.uint8)
                synt_s = rescale_intensity(artefact_gen.transform(norm_s), out_range=np.uint8)

                if SAVE_SYNTH_IMAGES:
                    path = os.path.join(synth_store_path, f'mri_{i}_slice_{j}.jpg')
                    skimage.io.imsave(path, synt_s)

                #--------------------------------------------------
                # Extract the features from both the artefact-free
                # image and the generated image.
                #--------------------------------------------------

                norm_i_features = fe_utils.extract_features_i(norm_s)
                norm_k_features = fe_utils.extract_features_k(norm_s)
                norm_g_features = fe_utils.extract_features_g(norm_s, enc, gen, critic, imshape_gan)
                norm_r_features = fe_utils.extract_features_r(norm_s, resnet, imshape_gan)

                synt_i_features = fe_utils.extract_features_i(synt_s)
                synt_k_features = fe_utils.extract_features_k(synt_s)
                synt_g_features = fe_utils.extract_features_g(synt_s, enc, gen, critic, imshape_gan)
                synt_r_features = fe_utils.extract_features_r(synt_s, resnet, imshape_gan)

                #--------------------------------------------------
                # Apply feature selection.
                #--------------------------------------------------

                norm_features = feature_selector(norm_i_features,
                                                 norm_k_features,
                                                 norm_g_features,
                                                 norm_r_features)

                synt_features = feature_selector(synt_i_features,
                                                 synt_k_features,
                                                 synt_g_features,
                                                 synt_r_features)

                norm_slices_features.append(norm_features)
                synt_slices_features.append(synt_features)

            #--------------------------------------------------
            # Concatenate the features extracted from the 
            # slices and add them to the dataset.
            #--------------------------------------------------

            norm_features = np.concatenate(norm_slices_features, axis=1)
            data_X.append(norm_features)
            data_y.append(0)
            
            synt_features = np.concatenate(synt_slices_features, axis=1)
            data_X.append(synt_features)
            data_y.append(1)

        dataset = { 'data': np.vstack(data_X), 'targets': np.array(data_y) }
        pickle.dump(dataset, open(dataset_path, 'wb'))


    #--------------------------------------------------
    # Dataset has been created and stored. Is now time 
    # to train the SVM model.
    #--------------------------------------------------   

    dataset = pickle.load(open(dataset_path, 'rb'))

    X, y = dataset['data'], dataset['targets']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RAND_SEED)

    pca = None
    if ARTEFACT_NAME == 'nobrain':
        # For `no-brain` "artifact" we select the feature 
        # group "R". As specified in the paper, this group
        # of feature is reduced using PCA.
        pca = PCA(n_components=64)
        X_train = pca.fit_transform(X_train)
        X_test  = pca.transform(X_test)

    scaler = StandardScaler().fit(X_train)

    clf = svm.SVC(C=10, 
                  gamma='auto', 
                  kernel='rbf', 
                  class_weight='balanced', 
                  probability=True,
                  max_iter=-1, 
                  shrinking=True, 
                  tol=1e-6)
    
    clf.fit(scaler.transform(X_train), y_train)

    y_pred = clf.predict(scaler.transform(X_test))
    print(classification_report(y_test, y_pred))

    print('saving models.')
    models_path = os.path.join(OUTPUT_FOLDER, 'models', 'SVM')
    if not os.path.exists(models_path): os.makedirs(models_path)
    pickle.dump(clf, open(os.path.join(models_path, f'{ARTEFACT_NAME}_svm.pkl'), 'wb'))
    pickle.dump(scaler, open(os.path.join(models_path, f'{ARTEFACT_NAME}_scaler.pkl'), 'wb'))
    if pca is not None:
        pickle.dump(scaler, open(os.path.join(models_path, f'{ARTEFACT_NAME}_pca.pkl'), 'wb'))