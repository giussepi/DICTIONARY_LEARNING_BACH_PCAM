# -*- coding: utf-8 -*-
"""  """

import os
from collections import OrderedDict

import numpy as np
from gtorch_utils.constants import DB
from gtorch_utils.models.managers import ModelMGR
from gtorch_utils.models.perceptrons import Perceptron, MLP
from lcksvd.dksvd import DKSVD
from sklearn.metrics import accuracy_score
from torch import optim

import settings
from constants.constants import CodeType, ProcessImageOption, PCamLabel, PCamSubDataset, Label
from dl_models.fine_tuned_resnet_18.models import TransferLearningResnet18
from utils.datasets.bach import plot_n_first_json_images, MiniPatch, TrainValTestSplit, \
    RawImages, RandomFaces, SparseCodes, WholeImage, RescaleResize, BachTorchDataset, \
    SelectNRandomPatches, BachTorchNetDataset, BACHDataset
from utils.datasets.pcam import WholeImage as PCamWholeImage, HDF5_2_PNG, FormatProvidedDatasetSplits, \
    PCamTorchDataset
from utils.utils import load_codes


def main():
    """  """
    ###########################################################################
    #                      Example PatchCamelyon dataset                       #
    ###########################################################################

    # HDF5_2_PNG(only_center=True)()
    # FormatProvidedDatasetSplits()()
    # PCamWholeImage()()

    ###########################################################################
    #                          Example BACH dataset                           #
    ###########################################################################

    # RescaleResize(.0625)()
    # TrainValTestSplit()()

    # create ROI files
    # option 1
    # WholeImage()()
    # option 2
    # MiniPatch()()
    # SelectNRandomPatches(100)()

    ###########################################################################
    #                            General operations                           #
    ###########################################################################

    # plot_n_first_json_images(15, os.path.join(settings.OUTPUT_FOLDER, settings.TRAIN_FOLDER_NAME),
    #                          (9, 9), False, 'my_folder', False, True, remove_axes=False, dpi=100)

    # Feature extraction ######################################################
    # option 1: raw images
    # BATCH
    # ri = RawImages(process_method=ProcessImageOption.GRAYSCALE, label_class=Label, sub_datasets=DB)
    # PatchCamelyon
    # ri = RawImages(process_method=ProcessImageOption.GRAYSCALE, label_class=PCamLabel, sub_datasets=PCamSubDataset)
    # ri.create_datasets_for_LC_KSVD('my_raw_dataset.json')

    # option 2: using random faces
    # BATCH
    # randfaces = RandomFaces(img_height=512, img_width=512,
    #                         process_method=ProcessImageOption.GRAYSCALE, label_class=Label, sub_datasets=DB)
    # PatchCamelyon
    # randfaces = RandomFaces(img_height=32, img_width=32, process_method=ProcessImageOption.GRAYSCALE,
    #                         label_class=PCamLabel, sub_datasets=PCamSubDataset)
    # randfaces.create_datasets_for_LC_KSVD('randfaces_dataset.json')

    # option 3: using CNN codes
    #
    # # download and save the pre-trained resnet 18
    # model = TransferLearningResnet18(fine_tune=False)
    # model.save('resnet18_feature_extractor.pt')
    #
    # # Fine tune resnet18
    # model = TransferLearningResnet18(fine_tune=True)
    # model.training_data_plot_grid()
    # model.train(num_epochs=25)
    # model.save('fine_tuned_resnet18.pt')
    # model.visualize_model()
    # model.test()

    #
    # # Create CNN codes
    # model = TransferLearningResnet18(fine_tune=True)
    # model.load('fine_tuned_resnet18.pt')
    # model.create_datasets_for_LC_KSVD('my_cnn_dataset.json')

    # option 4: using sparse representations
    # for BACH
    # ri = SparseCodes(
    #     process_method=ProcessImageOption.GRAYSCALE, label_class=Label, sub_datasets=DB,
    #     sparse_coding=DKSVD.get_sparse_representations,
    #     sparse_coding_kwargs=dict(D=np.load('D.npy'), sparsitythres=15)
    # )
    # for PatchCamelyon
    # ri = SparseCodes(
    #     process_method=ProcessImageOption.GRAYSCALE, label_class=PCamLabel, sub_datasets=PCamSubDataset
    #     sparse_coding=DKSVD.get_sparse_representations,
    #     sparse_coding_kwargs=dict(D=np.load('D.npy'), sparsitythres=15)
    # )
    # ri.create_datasets_for_LC_KSVD('sparse_codes_dataset.json')

    # Load features extracted
    # Example loading raw features (created using RawImages)
    # test = load_codes('my_raw_dataset_test.json', type_=CodeType.RAW)
    # print(test['codes'].shape)
    # print(test['labels'].shape)

    # train = load_codes('my_raw_dataset_train.json', type_=CodeType.RAW)
    # print(train['codes'].shape)
    # print(train['labels'].shape)

    # valid = load_codes('my_raw_dataset_valid.json', type_=CodeType.RAW)
    # print(valid['codes'].shape)
    # print(valid['labels'].shape)


if __name__ == '__main__':
    main()

    # Using extracted feature with several algorithms/models

    ###########################################################################
    #                                 Resnet18                                #
    ###########################################################################
    # # download and save the pre-trained resnet 18
    # model = TransferLearningResnet18(fine_tune=False)
    # model.save('resnet18_feature_extractor.pt')

    # # fine tunning
    # model = TransferLearningResnet18(fine_tune=True)
    # model.training_data_plot_grid()
    # model.train(num_epochs=25)
    # model.save('fine_tuned_resnet18.pt')
    # model.visualize_model()
    # model.test()

    # # Load a resnet18 as a fixed feature extractor
    # model2 = TransferLearningResnet18(fine_tune=False)
    # model2.load('resnet18_feature_extractor.pt')
    # # model2.visualize_model()
    # model2.test()

    # # Creating CNN codes
    # model = TransferLearningResnet18(fine_tune=True)
    # model.load('fine_tuned_resnet18.pt')
    # model.create_datasets_for_LC_KSVD('mydataset.json')

    # # LOAD CODES
    # test = load_codes('mydataset_test.json', type_=CodeType.CNN)
    # print(test['codes'].shape)
    # print(test['labels'].shape)

    # train = load_codes('mydataset_train.json', type_=CodeType.CNN)
    # print(train['codes'].shape)
    # print(train['labels'].shape)

    # val = load_codes('mydataset_val.json', type_=CodeType.CNN)
    # print(train['codes'].shape)
    # print(train['labels'].shape)

    ###########################################################################
    #                                 LC-KSVD1                                #
    ###########################################################################
    # test = load_codes('my_raw_dataset_test.json', type_=CodeType.RAW)
    # train = load_codes('my_raw_dataset_train.json', type_=CodeType.RAW)
    # val = load_codes('my_raw_dataset_val.json', type_=CodeType.RAW)

    # SPARSITYTHRES = 15
    # lcksvd = DKSVD(
    #     sparsitythres=SPARSITYTHRES, dictsize=train['labels'].shape[0]*SPARSITYTHRES,
    #     sqrt_alpha=.0012, timeit=True
    # )
    # Dinit, Tinit_T, Winit_T, Q = lcksvd.initialization4LCKSVD(*train.values())
    # np.save('Dinit.npy', Dinit, False)
    # np.save('Tinit_T.npy', Tinit_T, False)
    # np.save('Winit_T.npy', Winit_T, False)
    # np.save('Q.npy', Q, False)
    # D, X, T, W = lcksvd.labelconsistentksvd1(train['codes'], Dinit, train['labels'], Q, Tinit_T)
    # np.save('D.npy', D, False)
    # np.save('X.npy', X, False)
    # np.save('T.npy', T, False)
    # np.save('W.npy', W, False)
    # predictions, gamma = lcksvd.classification(D, W, test['codes'])
    # print('\nFinal recognition rate for LC-KSVD1 is : {0:.4f}'.format(
    #     accuracy_score(np.argmax(test['labels'], axis=0), predictions)))

    ###########################################################################
    #                                 LC-KSVD2                                #
    ###########################################################################
    # test = load_codes('my_raw_dataset_test.json', type_=CodeType.RAW)
    # train = load_codes('my_raw_dataset_train.json', type_=CodeType.RAW)
    # val = load_codes('my_raw_dataset_val.json', type_=CodeType.RAW)

    # SPARSITYTHRES = 15
    # lcksvd = DKSVD(
    #     sparsitythres=SPARSITYTHRES, dictsize=train['labels'].shape[0]*SPARSITYTHRES, timeit=True,
    #     sqrt_alpha=.0012, sqrt_beta=.0012, tol=1e-6, iterations=50, iterations4ini=20
    # )
    # Dinit, Tinit_T, Winit_T, Q = lcksvd.initialization4LCKSVD(*train.values())
    # np.save('Dinit.npy', Dinit, False)
    # np.save('Tinit_T.npy', Tinit_T, False)
    # np.save('Winit_T.npy', Winit_T, False)
    # np.save('Q.npy', Q, False)

    # Dinit = np.load('Dinit.npy')
    # Tinit_T = np.load('Tinit_T.npy')
    # Winit_T = np.load('Winit_T.npy')
    # Q = np.load('Q.npy')
    # D, X, T, W = lcksvd.labelconsistentksvd2(train['codes'], Dinit, train['labels'], Q, Tinit_T, Winit_T)
    # np.save('D.npy', D, False)
    # np.save('X.npy', X, False)
    # np.save('T.npy', T, False)
    # np.save('W.npy', W, False)
    # predictions, gamma = lcksvd.classification(D, W, test['codes'])
    # print('\nFinal recognition rate for LC-KSVD2 is : {0:.4f}'.format(
    #     accuracy_score(np.argmax(test['labels'], axis=0), predictions)))

    ###########################################################################
    #                                  D-KSVD                                 #
    ###########################################################################
    # test = load_codes('my_raw_dataset_test.json', type_=CodeType.RAW)
    # train = load_codes('my_raw_dataset_train.json', type_=CodeType.RAW)
    # val = load_codes('my_raw_dataset_val.json', type_=CodeType.RAW)

    # SPARSITYTHRES = 15
    # lcksvd = DKSVD(
    #     sparsitythres=SPARSITYTHRES, dictsize=train['labels'].shape[0]*SPARSITYTHRES,
    #     timeit=True
    # )
    # Dinit, Winit = lcksvd.initialization4DKSVD(*train.values())
    # predictions, gamma = lcksvd.classification(Dinit, Winit, train['codes'])
    # print('\nFinal recognition rate for D-KSVD is : {0:.4f}'.format(
    #     accuracy_score(np.argmax(train['labels'], axis=0), predictions)))

    ###########################################################################
    #                                Perceptron                               #
    ###########################################################################
    # test = load_codes('sparse_codes_dataset_test.json', type_=CodeType.SPARSE)
    # test = load_codes('my_raw_dataset_test.json', type_=CodeType.RAW)

    # ModelMGR(
    #     cuda=True,
    #     model=Perceptron(test['codes'].shape[0], test['labels'].shape[0]),
    #     sub_datasets=DB,  # PCamSubDataset
    #     dataset=BachTorchDataset,  # PCamTorchDataset
    #     dataset_kwargs=dict(filename_pattern='my_raw_dataset.json', code_type=CodeType.RAW),
    #     batch_size=6,
    #     shuffe=False,
    #     num_workers=16,
    #     optimizer=optim.SGD,
    #     optimizer_kwargs=dict(lr=1e-3, momentum=.9),
    #     lr_scheduler=None,
    #     lr_scheduler_kwargs={},
    #     epochs=600,
    #     earlystopping_kwargs=dict(min_delta=1e-5, patience=15),
    #     checkpoints=False,
    #     checkpoint_interval=5,
    #     checkpoint_path=OrderedDict(directory_path='tmp', filename=''),
    #     saving_details=OrderedDict(directory_path='tmp', filename='best_model.pth'),
    #     tensorboard=True
    # )()

    ###########################################################################
    #                          Multi-layer Perceptron                          #
    ###########################################################################
    # test = load_codes('my_raw_dataset_test.json', type_=CodeType.RAW)

    # ModelMGR(
    #     cuda=True,
    #     model=MLP(
    #         test['codes'].shape[0], test['codes'].shape[0],
    #         test['labels'].shape[0], dropout=.25, sigma=.1
    #     ),
    #     sub_datasets=DB,  # PCamSubDataset
    #     dataset=BachTorchDataset,  # PCamTorchDataset
    #     dataset_kwargs=dict(filename_pattern='my_raw_dataset.json', code_type=CodeType.RAW),
    #     batch_size=6,
    #     shuffe=False,
    #     num_workers=16,
    #     optimizer=optim.SGD,
    #     optimizer_kwargs=dict(lr=1e-4, momentum=.9),
    #     lr_scheduler=None,
    #     lr_scheduler_kwargs={},
    #     epochs=200,
    #     earlystopping_kwargs=dict(min_delta=1e-6, patience=15),
    #     checkpoints=False,
    #     checkpoint_interval=5,
    #     checkpoint_path=OrderedDict(directory_path='tmp'),
    #     saving_details=OrderedDict(directory_path='tmp', filename='best_model.pth'),
    #     tensorboard=True
    # )()

    ###########################################################################
    #                           Fine-tuned Resnet18                           #
    ###########################################################################
    # Train by reading images from disk
    # model = TransferLearningResnet18(fine_tune=True)
    # Train by reading extracted codes
    # model = TransferLearningResnet18(
    #     fine_tune=True, dataset_handler=BachTorchNetDataset,
    #     dataset_kwargs=dict(
    #         code_type=CodeType.SPARSE,
    #         filename_pattern='sparse_codes_dataset.json',
    #         original_shape=(1, 60)
    #     )
    # )
    # model.training_data_plot_grid()
    # model.train(num_epochs=25)
    # model.save('fine_tuned_resnet18.pt')
    # model.visualize_model()
    # model.test()
