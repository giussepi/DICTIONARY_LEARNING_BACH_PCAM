# -*- coding: utf-8 -*-
"""  """

import numpy as np
from sklearn.metrics import accuracy_score

# from dl_models.fine_tuned_resnet_18.models import TransferLearningResnet18
from dl_algorithms.lc_ksvd.dksvd import DKSVD
# import settings
# from utils.datasets.bach import plot_n_first_json_images, MiniPatch, TrainTestSplit
from utils.utils import load_cnn_codes


def main():
    """  """
    # TODO: the first we need to perform the train/test split and then create the minipatches

    # plot_n_first_json_images(5, os.path.join(settings.OUTPUT_FOLDER, settings.TRAIN_FOLDER_NAME),
    #                          (9, 9), False, 'my_folder', False, True)
    # plot_n_first_json_images(5, os.path.join(settings.OUTPUT_FOLDER, settings.TRAIN_FOLDER_NAME),
    #                          (9, 9), True, 'my_folder', True, True)
    # TrainTestSplit()()
    # MiniPatch()()

    # faces projected to 504 dimensional vector, original crops 192x168 (32256)

    # I need 512 features vectors to be used by LK-KSVD
    # I'll perform feature extraction using Resnet18

    # data['featureMat'].shape (504 vector, 2414 images)
    # data['labelMat'].shape (38 persons, 2414 images)

    # H_train = sp.zeros((int(labels.max()), training_feats.shape[1]), dtype=float)
    # for c in range(int(labels.max())):
    #     H_train[c, labels == (c+1)] = 1.


if __name__ == '__main__':
    main()
    # model = TransferLearningResnet18(fine_tune=True)
    # model.training_data_plot_grid()
    # model.train(num_epochs=25)
    # model.save('restnet18_feature_extractor_2.pt')
    # model.load('weights/resnet18_feature_extractor.pt')
    # model.load('weights/resnet18_fine_tuned_2.pt')
    # model.test()
    # model.visualize_model()

    # model.create_datasets_for_LC_KSVD('attempt3.json')
    test = load_cnn_codes('attempt3_test.json')
    # test['cnn_codes'].shape  # (512, 2100)
    # test['labels'].shape  # (4, 2100)

    train = load_cnn_codes('attempt3_train.json')
    # train['cnn_codes'].shape  # (512, 11900)
    # train['labels'].shape  # (4, 11900)

    ###########################################################################
    #                                 LC-KSVD1                                #
    ###########################################################################
    # lcksvd = DKSVD(dictsize=570, timeit=True)
    # Dinit, Tinit_T, Winit_T, Q = lcksvd.initialization4LCKSVD(*train.values())
    # np.save('Dinit.npy', Dinit, False)
    # np.save('Tinit_T.npy', Tinit_T, False)
    # np.save('Winit_T.npy', Winit_T, False)
    # np.save('Q.npy', Q, False)
    # D, X, T, W = lcksvd.labelconsistentksvd1(train['cnn_codes'], Dinit, train['labels'], Q, Tinit_T)
    # np.save('D.npy', D, False)
    # np.save('X.npy', X, False)
    # np.save('T.npy', T, False)
    # np.save('W.npy', W, False)
    # predictions, gamma = lcksvd.classification(D, W, test['cnn_codes'])
    # print('\nFinal recognition rate for LC-KSVD1 is : {0:.4f}'.format(
    #     accuracy_score(np.argmax(test['labels'], axis=0), predictions)))

    ###########################################################################
    #                                 LC-KSVD2                                #
    ###########################################################################
    # lcksvd = DKSVD(dictsize=570, timeit=True)
    # # Dinit, Tinit_T, Winit_T, Q = lcksvd.initialization4LCKSVD(*train.values())
    # # np.save('Dinit.npy', Dinit, False)
    # # np.save('Tinit_T.npy', Tinit_T, False)
    # # np.save('Winit_T.npy', Winit_T, False)
    # # np.save('Q.npy', Q, False)

    # Dinit = np.load('Dinit.npy')
    # Tinit_T = np.load('Tinit_T.npy')
    # Winit_T = np.load('Winit_T.npy')
    # Q = np.load('Q.npy')
    # D, X, T, W = lcksvd.labelconsistentksvd2(train['cnn_codes'], Dinit, train['labels'], Q, Tinit_T, Winit_T)
    # np.save('D_2.npy', D, False)
    # np.save('X_2.npy', X, False)
    # np.save('T_2.npy', T, False)
    # np.save('W_2.npy', W, False)
    # predictions, gamma = lcksvd.classification(D, W, test['cnn_codes'])
    # print('\nFinal recognition rate for LC-KSVD2 is : {0:.4f}'.format(
    #     accuracy_score(np.argmax(test['labels'], axis=0), predictions)))

    ###########################################################################
    #                                  D-KSVD                                 #
    ###########################################################################
    lcksvd = DKSVD(dictsize=570, timeit=True)
    Dinit, Winit = lcksvd.initialization4DKSVD(*train.values())
    predictions, gamma = lcksvd.classification(Dinit, Winit, train['cnn_codes'])
    print('\nFinal recognition rate for D-KSVD is : {0:.4f}'.format(
        accuracy_score(np.argmax(train['labels'], axis=0), predictions)))
