import os
import numpy as np

import SimpleITK as sitk
import sys
import datetime
import csv
import time
import keras.backend as K
import matplotlib.image as mpimage
from sklearn.utils import shuffle
import tensorflow as tf
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization, InputSpec
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
import model
import glob

class train():

    def __init__(self):


        self.test_path = False
        self.test_epoch = False
        self.model_dir = 'G:\Model\HDAI'
        self.retrain_path = False  #
        self.retrain_epoch = False  #

        self.save_validation_loss = True

        self.augmentation = True
        #self.augmentation_orientation = 'axial'

        self.patch = False
        # self.patchA_size = 64
        # self.strideA_size = 32
        # self.patchB_size = 64
        # self.strideB_size = 32

        self.epochs =400  # choose multiples of 20 since the models are saved each 20th epoch
        self.use_linear_decay = True#False # Linear decay of learning rate, for both discriminators and generators
        self.decay_epoch = 100#False #11#101 # The epoch where the linear decay of the learning rates start
        self.batch_size = 1  # Number of volumes per batch
        self.iter_perEpoch = 1000# 220

        self.model = None
        self.data = None


     # Tweaks

        self.save_training_vol = True  # Save or not example training results or only tmp.png
        self.save_models = True  # Save or not the generator and discriminator models
        self.tmp_vol_update_frequency = 3
        self.train_iterations = 1

    def make_folders(self):

        # ===== Folders and configuration =====
        self.date_time = time.strftime('%Y%m%d-%H%M%S', time.localtime())
        self.out_dir = os.path.join(self.model_dir, self.date_time)


        if self.retrain_path :
            self.out_dir =  os.path.join(self.model_dir,self.retrain_path)

        if self.test_path:
            self.out_dir =  os.path.join(self.model_dir, self.test_path)

        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        if self.save_training_vol:
            self.out_dir_volumes = os.path.join(self.out_dir, 'training_volumes')
            if not os.path.exists(self.out_dir_volumes):
                os.makedirs(self.out_dir_volumes)

        if self.save_models:
            self.out_dir_models = os.path.join(self.out_dir, 'models')
            self.out_dir_train_losses = os.path.join(self.out_dir, 'train_losses')
            if not os.path.exists(self.out_dir_models):
                os.makedirs(self.out_dir_models)
            if not os.path.exists(self.out_dir_train_losses):
                os.makedirs(self.out_dir_train_losses)

            if self.save_validation_loss:
                self.out_dir_validation_losses = os.path.join(self.out_dir, 'validation_losses')
                if not os.path.exists(self.out_dir_validation_losses):
                    os.makedirs(self.out_dir_validation_losses)

    def load_data(self, data):
        self.data = data

    def load_model(self, model):
        self.model = model

    def train(self):

        def run_training_batch():
            # ======= Generator training ==========
            synthetic_volumes_B= self.model.base_model.predict(real_volumes_A)

            target_data = []
            for _ in range(len(self.model.compile_weights)) :
                target_data.append(real_volumes_B)

            # Train on batch
            for _ in range(self.train_iterations):
                loss.append(self.model.model.train_on_batch(x=[real_volumes_A], y=target_data))

            # =====================================
            
            # Update learning rates
            if self.use_linear_decay and epoch >= self.decay_epoch:
                self.update_lr(self.model.model, decay, loop_index, epoch)


            losses.append(loss[-1])

            # Print training status
            print('\n')
            print('Epoch ---------------------', epoch, '/', self.epochs)
            print('Loop index ----------------', loop_index_idx, '/',  # ) * loop_index_idx
                  nr_vol_per_epoch)  # * len(self.trainA_file_names))
            print('  Summary:')
            print('lr', K.get_value(self.model.model.optimizer.lr))
            print('loss: ', loss[-1][0])

            idx = 0
            for compile_loss in self.model.compile_losses :
                idx = idx + 1
                try:
                    print(compile_loss.name, loss[-1][idx])
                except:
                    print('additional_loss' + str(idx + 1) + ':', loss[-1][idx])


            self.print_ETA(start_time, epoch, nr_vol_per_epoch, loop_index_idx)
            sys.stdout.flush()

            if loop_index % self.tmp_vol_update_frequency * self.batch_size == 0:
                # Save temporary images continously
                self.save_tmp_images(real_volumes_A[0][:,:,0], synthetic_volumes_B[0][:,:,0], real_volumes_B[0][:,:,0])

        # ======================================================================
        # Begin training
        # ======================================================================
        if self.save_training_vol:
            if not os.path.exists(os.path.join(self.out_dir_volumes, 'train_A')):
                os.makedirs(os.path.join(self.out_dir_volumes, 'train_A'))
                os.makedirs(os.path.join(self.out_dir_volumes, 'test_A'))


        losses = []

        # Start stopwatch for ETAs
        start_time = time.time()
        timer_started = False

        if self.retrain_epoch:
            start = self.retrain_epoch + 1
        else:
            start = 1

        for epoch in range(start, self.epochs + 1):

            loss = []
            loop_index_idx = 1

            trainA_file_name, trainB_file_name = shuffle(self.data.trainA_file_names, self.data.trainB_file_names)
            trainA_file_name = trainA_file_name[: self.iter_perEpoch]
            trainB_file_name = trainB_file_name[: self.iter_perEpoch]

            A_train, B_train = self.data.load_trainset(trainA_file_name, trainB_file_name)

            if self.augmentation:
                A_train, B_train = self.data_augmentation2D(A_train, B_train, epoch)

            if self.patch:
                A_train = self.extract_volume_patches(A_train, self.patchA_size, self.strideA_size)
                B_train = self.extract_volume_patches(B_train, self.patchB_size, self.strideA_size)

            # Linear learning rate decay
            if self.use_linear_decay:
                decay = self.get_lr_linear_decay_rate()

            nr_train_vol = self.iter_perEpoch
            nr_vol_per_epoch = int(np.ceil(nr_train_vol / self.batch_size) * self.batch_size)

            random_order = np.concatenate((np.random.permutation(nr_train_vol),
                                           np.random.randint(nr_train_vol, size=nr_vol_per_epoch - nr_train_vol)))

            # Train on volume batch
            for loop_index in range(0, nr_vol_per_epoch, self.batch_size):

                indices = random_order[loop_index:loop_index + self.batch_size]

                real_volumes_A = A_train[indices]
                real_volumes_B = B_train[indices]

                # Train on volume batch
                run_training_batch()

                loop_index_idx += self.batch_size

                # Start timer after first (slow) iteration has finished
                if not timer_started:
                    start_time = time.time()
                    timer_started = True

            # Save training volumes
            # if self.save_training_vol and epoch % self.save_training_vol_interval == 0:
            print('\n', '\n', '-------------------------Saving volumes for epoch', epoch,
                  '-------------------------',
                  '\n', '\n')
            self.save_epoch_images(epoch)
            self.save_model(self.model.model, epoch)

            # # Save training history
            # training_history = {
            #     'DB_losses': D_B_losses,
            #     'G_AB_adversarial_losses': G_AB_adversarial_losses,
            #     'G_AB_supervised_losses': G_AB_supervised_losses,
            #     'G_AB_gd_losses' : G_AB_gd_losses,
            #     'G_losses': G_losses
            # }
            #self.write_loss_data_to_file(training_history)

            print('..........save train loss...........')


            train_losses = {}
            idx = 0
            train_losses['total_loss'] = loss[-1][idx].mean()
            for compile_loss in self.model.compile_losses:
                idx = idx + 1
                try:
                    train_losses[compile_loss.name] = loss[-1][idx].mean()
                except:
                    train_losses['additional_loss' + str(idx + 1)] = loss[-1][idx].mean()



            self.write_loss_data_by_epoch(train_losses, epoch, self.out_dir_train_losses)



    # ===============================================================================
    # Learning rates
    def get_lr_linear_decay_rate(self):
        # Calculate decay rates
        # max_nr_volumes = max(len(self.A_train), len(self.B_train))
        nr_batches_per_epoch = int(np.ceil(self.iter_perEpoch / self.batch_size))

        updates_per_epoch = nr_batches_per_epoch
        nr_decay_updates = (self.epochs - self.decay_epoch + 1) * updates_per_epoch
        decay = self.model.learning_rate / nr_decay_updates
        return decay
    
    def update_lr(self, model, decay, loop_index, epoch):
        new_lr = K.get_value(model.optimizer.lr) - decay
        if new_lr < 0:
            new_lr = 0
        # print(K.get_value(model.optimizer.lr))
        K.set_value(model.optimizer.lr, new_lr)

        if loop_index == 0:
            lr_path = '{}/test_A/epoch{}_lr_{}.npy'.format(self.out_dir_volumes, epoch, loop_index)
            f = open(lr_path, mode='wt', encoding='utf-8')
            f.write(str(new_lr))
            f.close()

    # ===============================================================================
    # save datas
    def join_and_save(self, images, save_path):
        # Join images
        image = np.hstack(images)
        # Save images
        #mpimage.imsave(save_path, image, vmin=-1, vmax=1, cmap='gray')
        mpimage.imsave(save_path, image, cmap='gray')

    def save_tmp_images(self, real_image_A, synthetic_image_B, real_image_B ):
        try:
            # Add dimensions if A and B have different number of channels
            # if self.channels_A == 1 and self.channels_B == 3:
            # real_image_A = np.tile(real_image_A, [1,1,3])
            # synthetic_image_B = np.tile(synthetic_image_B, [1, 1, 3])
            # real_image_B= np.tile(real_image_B, [1, 1, 3])
            # elif self.channels_B == 1 and self.channels_A == 3:
            #     synthetic_image_B = np.tile(synthetic_image_B, [1,1,3])
            #     real_image_B = np.tile(real_image_B, [1,1,3])

            save_path = '{}/tmp.png'.format(self.out_dir)
            self.join_and_save((real_image_A, synthetic_image_B, real_image_B), save_path)


        except: # Ignore if file is open
            pass

    def save_epoch_images(self, epoch, num_saved_images=1):

        def jaccard(x, y):
            y = np.where(y<0.5, 0 ,1)

            x = np.asarray(x, np.bool)  # Not necessary, if you keep your data
            y = np.asarray(y, np.bool)  # in a boolean array already!
            return np.double(np.bitwise_and(x, y).sum()) / np.double(np.bitwise_or(x, y).sum())

        def dice(im1, im2):
            im2 = np.where(im2 < 0.5, 0, 1)
            im1 = np.asarray(im1).astype(np.bool)
            im2 = np.asarray(im2).astype(np.bool)
            # Compute Dice coefficient
            intersection = np.logical_and(im1, im2)
            return 2. * intersection.sum() / (im1.sum() + im2.sum())

        # Save training images
        testA_file_name = self.data.testA_file_names  # shuffle(self.data.testA_file_names)
        #testA_file_name = testA_file_name[: 1]
        testB_file_name = self.data.testB_file_names  # shuffle(self.data.testA_file_names)
        #testB_file_name = testB_file_name[: 1]
        A_test, B_test = self.data.load_testset(testA_file_name, testB_file_name)

        nr_train_im = A_test.shape[0]

        A2C_dsc_list= []
        A2C_ji_list = []

        A4C_dsc_list = []
        A4C_ji_list = []

        #
        #rand_ind = np.random.randint(nr_train_im)

        save_path = '{}/test_A/epoch{}'.format(self.out_dir_volumes, epoch)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        i = 0
        for nr_ind in range( nr_train_im ):
            #

            real_image_A = A_test[nr_ind][:,:,0]
            real_image_B = B_test[nr_ind][:,:,0]
            synthetic_image_B = self.model.base_model.predict(A_test)[nr_ind][:,:,0]

            if self.data.testA_file_names[nr_ind].split('\\')[-1].split(".")[0].split("_")[-1] == 'A2C':
                A2C_dsc_list.append(dice(real_image_B, synthetic_image_B))
                A2C_ji_list.append(jaccard(real_image_B, synthetic_image_B))
            else:
                A4C_dsc_list.append(dice(real_image_B, synthetic_image_B))
                A4C_ji_list.append(jaccard(real_image_B, synthetic_image_B))
            # Add dimensions if A and B have different number of channels
            # if self.data.channels_A == 1 and sel.data.channels_B == 3:
            #     real_image_A = np.tile(real_image_A, [1, 1, 3])
            # elif self.data.channels_B == 1 and self.data.channels_A == 3:
            #     synthetic_image_B = np.tile(synthetic_image_B, [1, 1, 3])
            #     real_image_B = np.tile(real_image_B, [1, 1, 3])

            img_path = save_path +'/{}'.format(self.data.testA_file_names[nr_ind].split('\\')[-1])

            #real_image_A = np.tile(real_image_A, [1, 1, 3])

            if i == 0 :
                self.join_and_save((real_image_A, synthetic_image_B, real_image_B), img_path)

            i = i + 1



        f = open(save_path+'//A2Cdsc.txt', 'w')
        f.write("%f %f" % (np.mean(np.array(A2C_dsc_list)), np.std(np.array(A2C_dsc_list)) ))
        f.close()

        f = open(save_path+'//A2CJAC.txt', 'w')
        f.write("%f %f" %  ( np.mean(np.array(A2C_ji_list)), np.std(np.array(A2C_ji_list)) ))
        f.close()

        f = open(save_path + '//A4Cdsc.txt', 'w')
        f.write("%f %f" % (np.mean(np.array(A4C_dsc_list)), np.std(np.array(A4C_dsc_list))))
        f.close()

        f = open(save_path + '//A4CJAC.txt', 'w')
        f.write("%f %f" % (np.mean(np.array(A4C_ji_list)), np.std(np.array(A4C_ji_list))))
        f.close()


    def save_epoch_volumes(self, epoch, num_saved_volumes=1):

        testA_file_name = self.data.testA_file_names#shuffle(self.data.testA_file_names)
        testA_file_name = testA_file_name[: 1]


        A_test, A_test_image = self.data.load_testset(testA_file_name)


        if self.patch:

            real_volume_A = self.extract_volume_patches(A_test[0], self.patchA_size, self.strideA_size)
            synthetic_volume_B = []
            for i in range(real_volume_A.shape[0]):
                synthetic_volume_B.extend(self.cGAN.G_A2B.predict(real_volume_A[i][np.newaxis, :, :, :, :]))
            synthetic_volume_B = np.array(synthetic_volume_B)
        else:
            #print(A_test_image.shape)
            synthetic_volume_B = np.array(self.cGAN.G_A2B.predict(A_test[0][np.newaxis, :, :, :, :]))

        refer_img = A_test_image[0]

        save_path_syntheticB = '{}/test_A/epoch{}_syntheticB.nii.gz'.format(self.out_dir_volumes, epoch)

        # rescale to original range
        if self.data.tanh_norm:
            synthetic_volume_B = (synthetic_volume_B + 1) / 2

        synthetic_volume_B = synthetic_volume_B * (self.data.inputB_max - self.data.inputB_min)
        synthetic_volume_B = synthetic_volume_B + self.data.inputB_min


        new_img = sitk.GetImageFromArray(synthetic_volume_B)
        new_img.SetSpacing(refer_img.GetSpacing())
        new_img.SetDirection(refer_img.GetDirection())
        new_img.SetOrigin(refer_img.GetOrigin())

        sitk.WriteImage(new_img, save_path_syntheticB)


    def save_model(self, model, epoch):

        weights_path = '{}/{}_epoch_{}.hdf5'.format(self.out_dir_models, model.name, epoch)
        model.save_weights(weights_path)

        model_path = '{}/{}_epoch_{}.json'.format(self.out_dir_models, model.name, epoch)
        model_json_string = model.to_json()
        with open(model_path, 'w') as outfile:
            outfile.write(model_json_string)
        print('{} has been saved in saved_models/{}/'.format(model.name, self.date_time))

    def write_loss_data_by_epoch(self, losses, epoch , path):
        keys = losses.keys()
        with open(path + '/' + '{:03d}.csv'.format(epoch), 'w') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames = keys)
            writer.writeheader()
            writer.writerow(losses)
            #$writer.writerow(*[losses[key] for key in keys])

    # def dsc(self, pred, true, k=1):
    #
    #     intersection = np.sum(pred[true == k]) * 2.0
    #     dice = intersection / (np.sum(pred) + np.sum(true))
    #
    #     return dice

    def test(self):

        # load generator A to B
        path_to_models = os.path.join(self.model_dir, self.test_path, 'models', 'G_A2B_model_epoch_{}.json'.format(self.test_epoch) )
        path_to_weights = os.path.join(self.model_dir, self.test_path, 'models', 'G_A2B_model_epoch_{}.hdf5'.format(self.test_epoch) )

        json_file = open(path_to_models, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        generator_model = model_from_json(loaded_model_json,  custom_objects={'ReflectionPadding3D': cGAN.ReflectionPadding3D, 'InstanceNormalization': InstanceNormalization})

        #generator_model = tf.keras.models.load_model(path_to_models)
        generator_model.load_weights(path_to_weights)

        real_path = os.path.join(self.data.data_root, self.data.volume_folder, 'realA')
        if not os.path.exists(real_path):
            os.mkdir(real_path)

        syn_path = os.path.join(self.data.data_root, self.data.volume_folder, 'syntheticB')
        if not os.path.exists(syn_path):
            os.mkdir(syn_path )

        A_test, A_test_image = self.data.load_testset(self.data.testA_file_names )


        for i in range(A_test.shape[0]):
            if self.patch:
                real_volume_A = self.extract_volume_patches(A_test[i], self.patchA_size, self.strideA_size)
                synthetic_volume_B = []
                for j in range(real_volume_A.shape[0]):
                    synthetic_volume_B.extend(generator_model .predict(real_volume_A[j][np.newaxis, :, :, :, :]))

                real_volume_A = np.array(real_volume_A)
                synthetic_volume_B = np.array(synthetic_volume_B)

                real_volume_A = self.reconstruct_from_patches(real_volume_A[:, :, :, :, 0], A_test[i][:,:,:,0].shape, self.strideA_size)
                synthetic_volume_B = self.reconstruct_from_patches(synthetic_volume_B[:, :, :, :, 0], A_test[i][:,:,:,0].shape, self.strideB_size)



            else:
                # print(A_test_image.shape)
                synthetic_volume_B = np.array(generator_model.predict(A_test[i][np.newaxis, :, :, :, :]))
                real_volume_A = A_test[i][np.newaxis, :, :, :, :]

            refer_img = A_test_image[i]

            sub_name = self.data.testA_file_names[i].split("\\")[-1].split("T1.nii")[0]
            synt_path = os.path.join(syn_path, '{}_synthetic.nii'.format(sub_name))
            real_path = os.path.join(syn_path, '{}_real.nii'.format(sub_name))


            # rescale to original range
            if self.data.tanh_norm :
                synthetic_volume_B = (synthetic_volume_B + 1) / 2

            synthetic_volume_B = synthetic_volume_B * (self.data.inputB_max - self.data.inputB_min)
            synthetic_volume_B = synthetic_volume_B + self.data.inputB_min

            real_volume_A =  real_volume_A  * (self.data.inputA_max - self.data.inputA_min)
            real_volume_A  =  real_volume_A + self.data.inputA_min


            new_img = sitk.GetImageFromArray(synthetic_volume_B)
            new_img.SetSpacing(refer_img.GetSpacing())
            new_img.SetDirection(refer_img.GetDirection())
            new_img.SetOrigin(refer_img.GetOrigin())
            sitk.WriteImage(new_img, synt_path)

            real_img = sitk.GetImageFromArray(real_volume_A )
            real_img.SetSpacing(refer_img.GetSpacing())
            real_img.SetDirection(refer_img.GetDirection())
            real_img.SetOrigin(refer_img.GetOrigin())
            sitk.WriteImage(real_img, real_path)







    def retrain(self):

        # load generator A to B


#        path_to_weights = os.path.join(self.model_dir, self.retrain_path, 'models', 'G_A2B_model_epoch_{}.hdf5'.format(self.retrain_epoch))

        path_to_weights = glob.glob( os.path.join(self.model_dir, self.retrain_path, 'models', '*'+'{}.hdf5'.format(self.retrain_epoch)) )[0]
        print(path_to_weights)
        self.model.model.load_weights(path_to_weights)

        # load discriminator B
        # path_to_weights = os.path.join(self.model_dir, self.retrain_path, 'models',
        #                                'D_B_model_epoch_{}.hdf5'.format(self.retrain_epoch))
        # self.cGAN.D_B.load_weights(path_to_weights)

        self.train()


    # ===============================================================================
    # data_augmentation in 3d direction
    def data_augmentation3D(self, A_train, B_train, epoch):

        def get_slice_orientation(orientation, index, volume):

            if self.augmentation_orientation == 'axial':  # axial
                slice = volume[index, :, :]
            if self.augmentation_orientation == 'sagittal':  # sagittal
                slice = volume[:, :, index]
            if self.augmentation_orientation == 'coronal':  # coronal
                slice = volume[:, index, :]

            return slice

        source = A_train
        target = B_train

        augmented_source_list = []
        augmented_target_list = []

        aug_num = 1
        for aug in range(aug_num):

            random = np.random.randint(1, 4)

            self.orientation = 'axial'
            if random == 1:
                self.orientation = 'axial'
            if random == 2:
                self.orientation = 'coronal'
            if random == 3:
                self.orientation = 'sagittal'

            print(self.orientation)
            # subject num
            for i in range(source.shape[0]):
                # prepare iterator

                # create image data augmentation generator
                datagen = ImageDataGenerator(rotation_range=4.5, zoom_range=0.3, shear_range=0.04, horizontal_flip=True)
                # shear_range=0.03
                if self.orientation == 'sagittal':  # data should not be flipped when sagittal
                    datagen = ImageDataGenerator(rotation_range=4.5, zoom_range=0.3, shear_range=0.04)

                source_volume = source[i, :, :, :, :]
                target_volume = target[i, :, :, :, :]

                augmented_source = np.full((source.shape[1], source.shape[1], source.shape[1]), -1.0)
                augmented_target = np.full((source.shape[1], source.shape[1], source.shape[1]), -1.0)

                for j in range(source.shape[1]):

                    source_slice = get_slice_orientation(self.orientation, j, source_volume)
                    target_slice = get_slice_orientation(self.orientation, j, target_volume)

                    source_slice = source_slice[np.newaxis, :, :, :]  # , np.newaxis]
                    target_slice = target_slice[np.newaxis, :, :, :]  # , np.newaxis]

                    itA = datagen.flow(source_slice, batch_size=1, seed=epoch)
                    itB = datagen.flow(target_slice, batch_size=1, seed=epoch)

                    # generate batch of images
                    batchA = itA.next()
                    batchB = itB.next()


                    imageA = batchA[0, :, :, 0]
                    imageB = batchB[0, :, :, 0]

                    if self.orientation == 'axial':  # axial
                        augmented_source[j, :, :] = imageA
                        augmented_target[j, :, :] = imageB

                    if self.orientation == 'sagittal':  # sagittal
                        augmented_source[:, :, j] = imageA
                        augmented_target[:, :, j] = imageB

                    if self.orientation == 'coronal':  # coronal
                        augmented_source[:, j, :] = imageA
                        augmented_target[:, j, :] = imageB

                augmented_source_list.append(augmented_source[:, :, :, np.newaxis])
                augmented_target_list.append(augmented_target[:, :, :, np.newaxis])

        return (np.array(augmented_source_list), np.array(augmented_target_list))

    def data_augmentation2D(self, A_train, B_train, epoch):

        #source = A_train
        #target = B_train

        augmented_source_list = []
        augmented_target_list = []

        # create image data augmentation generator
        datagen = ImageDataGenerator(rotation_range=4.5, zoom_range=0.3, shear_range=0.04, horizontal_flip=True, width_shift_range=0.05, height_shift_range=0.05)
        # shear_range=0.03
        for i in range(A_train.shape[0]):

#            source_slice = source[np.newaxis, :, :]  # , np.newaxis]
 #           target_slice = target[np.newaxis, :, :]  # , np.newaxis]

            itA = datagen.flow(A_train[i][np.newaxis, :, :,:], batch_size=1, seed=epoch)
            itB = datagen.flow(B_train[i][np.newaxis, :, :,:], batch_size=1, seed=epoch)

            # generate batch of images
            batchA = itA.next()
            batchB = itB.next()

            imageA = batchA[0, :, :, 0]
            imageB = batchB[0, :, :, 0]

            augmented_source_list.append(imageA[:, :, np.newaxis])
            augmented_target_list.append(imageB[:, :, np.newaxis])

        return (np.array(augmented_source_list), np.array(augmented_target_list))

    # ===============================================================================
    # Patch extraction & reconstruction
    def extract_volume_patches(self, img_arr, size=64, stride=16):

        if size % stride != 0:
            raise ValueError("size % stride must be equal 0")

        patches_list = []
        if len(img_arr.shape) == 5:
            for i in range(img_arr.shape[0]):
                temp = self.get_3Dpatches(img_arr[i, :, :, :, 0], size, stride)
                patches_list.extend(temp)
            np.stack(patches_list)

        if len(img_arr.shape) == 4:
            patches_list = self.get_3Dpatches(img_arr[:, :, :, 0], size, stride)

        return np.array(patches_list)

    def get_3Dpatches(self, img_arr, size=64, stride=16):

        if size % stride != 0:
            raise ValueError("size % stride must be equal 0")

        patches_list = []
        overlapping = 0
        if stride != size:
            overlapping = (size // stride) - 1

        i_max = img_arr.shape[0] // stride - overlapping
        j_max = img_arr.shape[1] // stride - overlapping
        k_max = img_arr.shape[2] // stride - overlapping
        # print(i_max)

        for i in range(i_max):
            for j in range(j_max):
                for k in range(k_max):
                    # print(i*stride, i*stride+size, j*stride, j*stride+size ,k*stride, k*stride+size)

                    patches_list.append(
                        img_arr[
                        i * stride: i * stride + size,
                        j * stride: j * stride + size,
                        k * stride: k * stride + size
                        ]
                    )

        patches = np.stack(patches_list)
        return patches[:, :, :, :, np.newaxis]


    # ===============================================================================
    # Other output
    def print_ETA(self, start_time, epoch, nr_vol_per_epoch, loop_index):
        passed_time = time.time() - start_time

        iterations_so_far = ((epoch - 1) * nr_vol_per_epoch + loop_index) / self.batch_size
        iterations_total = self.epochs * nr_vol_per_epoch / self.batch_size
        iterations_left = iterations_total - iterations_so_far
        eta = round(passed_time / (iterations_so_far + 1e-5) * iterations_left)

        passed_time_string = str(datetime.timedelta(seconds=round(passed_time)))

        try:
            eta_string = str(datetime.timedelta(seconds=eta))
            print('Elapsed time', passed_time_string, ': ETA in', eta_string)
        except:
            print('Too long elapsed time')

class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            if len(image.shape) == 3:
                image = image[np.newaxis, :, :, :]

            if self.num_imgs < self.pool_size:  # fill up the image pool
                self.num_imgs = self.num_imgs + 1
                if len(self.images) == 0:
                    self.images = image
                else:
                    self.images = np.vstack((self.images, image))

                if len(return_images) == 0:
                    return_images = image
                else:
                    return_images = np.vstack((return_images, image))

            else:  # 50% chance that we replace an old synthetic image
                p = np.random.rand()
                if p > 0.5:
                    random_id = np.random.randint(0, self.pool_size)
                    tmp = self.images[random_id, :, :, :]
                    tmp = tmp[np.newaxis, :, :, :]
                    self.images[random_id, :, :, :] = image[0, :, :, :]
                    if len(return_images) == 0:
                        return_images = tmp
                    else:
                        return_images = np.vstack((return_images, tmp))
                else:
                    if len(return_images) == 0:
                        return_images = image
                    else:
                        return_images = np.vstack((return_images, image))

        return return_images

class volumePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_vols = 0
            self.volumes = []

    def query(self, volumes):
        if self.pool_size == 0:
            return volumes
        return_volumes = []
        for volume in volumes:
            if len(volume.shape) == 4:
                volume = volume[np.newaxis, :, :, :, :]

            if self.num_vols < self.pool_size:  # fill up the volume pool
                self.num_vols = self.num_vols + 1
                if len(self.volumes) == 0:
                    self.volumes = volume
                else:
                    self.volumes = np.vstack((self.volumes, volume))

                if len(return_volumes) == 0:
                    return_volumes = volume
                else:
                    return_volumes = np.vstack((return_volumes, volume))

            else:  # 50% chance that we replace an old synthetic volume
                p = np.random.rand()
                if p > 0.5:
                    random_id = np.random.randint(0, self.pool_size)
                    tmp = self.volumes[random_id, :, :, :, :]
                    tmp = tmp[np.newaxis, :, :, :, :]
                    self.volumes[random_id, :, :, :, :] = volume[0, :, :, :, :]
                    if len(return_volumes) == 0:
                        return_volumes = tmp
                    else:
                        return_volumes = np.vstack((return_volumes, tmp))
                else:
                    if len(return_volumes) == 0:
                        return_volumes = volume
                    else:
                        return_volumes = np.vstack((return_volumes, volume))

        return return_volumes