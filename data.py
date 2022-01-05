import os
import numpy as np
import glob
import cv2
import SimpleITK as sitk

class data():

    def __init__(self):
        # data location
        self.data_root = 'G:/data/Dental/'
        self.volume_folder = 'augmented'
        self.view = 'ALL' # 'A4C' 'BOTH'
        self.preprocess = False
        self.inputA_min = 0
        self.inputA_max = 2500
        self.inputB_min = -1000
        self.inputB_max = 2000
        self.tanh_norm = False

        self.inputA_size = (434, 636)
        self.inputA_channel = 1
        self.inputB_size = (434, 636)
        self.inputB_channel = 1
        self.padding_center = True # pad images if smaller than input size


        self.trainA_path = None
        self.trainA_path = None
        self.testA_path = None
        self.testB_path = None

        self.trainA_file_names = None
        self.trainB_file_names = None
        self.testA_file_names = None
        self.testB_file_names = None

    def load_datalist(self):

        # file paths
        # self.trainA_path = os.path.join(self.data_root, self.volume_folder, 'trainA_nii')
        # self.trainB_path = os.path.join(self.data_root, self.volume_folder,'trainB_nii')
        # self.testA_path = os.path.join(self.data_root, self.volume_folder, 'testA_nii')
        # self.testB_path = os.path.join(self.data_root, self.volume_folder, 'testB_nii')
        self.trainA_path = os.path.join(self.data_root, self.volume_folder, 'train', self.view)
        self.trainB_path = os.path.join(self.data_root, self.volume_folder,'train', self.view)
        self.testA_path = os.path.join(self.data_root, self.volume_folder, 'validation', self.view)
        self.testB_path = os.path.join(self.data_root, self.volume_folder, 'validation',self.view)

        # volume file names
        self.trainA_file_names = sorted(glob.glob(os.path.join(self.trainA_path, '*.png')))
        print(self.trainA_file_names)
        self.trainB_file_names = sorted(glob.glob(os.path.join(self.trainB_path, '*.npy')))
        self.testA_file_names = sorted(glob.glob(os.path.join(self.testA_path, '*.png')))
        self.testB_file_names = sorted(glob.glob(os.path.join(self.testB_path, '*.npy')))

    def padding_volume(self, volume, padding_value, volume_size):

        # create new image of desired size
        org_width = volume.shape[0]
        org_height = volume.shape[1]
        org_depth = volume.shape[2]

        padded_width = volume_size[0]
        padded_height = volume_size[1]
        padded_depth =volume_size[2]

        # compute center offset
        source_xx = (padded_width - org_width) // 2
        source_yy = (padded_height - org_height) // 2
        source_zz = (padded_depth - org_depth) // 2

        padded_volume = np.full((padded_width, padded_height, padded_depth), padding_value)

        # copy img image into center of result image
        if self.padding_center :
            padded_volume[source_xx:source_xx + org_width, source_yy:source_yy + org_height,
            source_zz:source_zz + org_depth] = volume
        else :
            # copy img image into top of result image
            padded_volume[0:org_width, 0:org_height, 0:org_depth] = volume

        return padded_volume

    def padding_slice(self, volume, padding_value, volume_size):

        # create new image of desired size
        org_width = volume.shape[0]
        org_height = volume.shape[1]
        #org_depth = volume.shape[2]

        padded_width = volume_size[0]
        padded_height = volume_size[1]
        #padded_depth = volume_size[2]

        # compute center offset
        source_xx = (padded_width - org_width) // 2
        source_yy = (padded_height - org_height) // 2
        #source_zz = (padded_depth - org_depth) // 2

        padded_volume = np.full((padded_width, padded_height), padding_value)

        # copy img image into center of result image
        if self.padding_center:
            padded_volume[source_xx:source_xx + org_width, source_yy:source_yy + org_height] = volume
        else:
            # copy img image into top of result image
            padded_volume[0:org_width, 0:org_height] = volume

        return padded_volume

    def load_trainset(self, listA, listB):

        source_slices = []
        target_slices = []

        for idx in range(len(listA)):
            source_slice = self.read_png(listA[idx])
            target_slice = self.read_npy(listB[idx])

            # padding image
            # if source_volume.shape[0] < self.inputA_size[0] or source_volume.shape[1] < self.inputA_size[1] or \
            #         source_volume.shape[2] < self.inputA_size[2]:
            #     source_volume = self.padding_volume(source_volume, self.inputA_min, self.inputA_size)
            #
            # if target_volume.shape[0] < self.inputB_size[0] or target_volume.shape[1] < self.inputB_size[1] or \
            #         target_volume.shape[2] < self.inputB_size[2]:
            #     target_volume = self.padding_volume(target_volume, self.inputB_min, self.inputB_size)
            # if source_slice.shape[0] < self.inputA_size[0] or source_slice.shape[1] < self.inputA_size[1] :
            #     source_slice = self.padding_slice(source_slice, self.inputA_min, self.inputA_size)
            # if target_slice.shape[0] < self.inputB_size[0] or target_slice.shape[1] < self.inputB_size[1]:
            #     target_slice = self.padding_slice(source_slice, self.inputB_min, self.inputB_size)

            source_slice = cv2.resize(source_slice, (self.inputA_size[0], self.inputA_size[1]), interpolation=cv2.INTER_AREA)
            target_slice = cv2.resize(target_slice, (self.inputB_size[0], self.inputB_size[1]), interpolation=cv2.INTER_AREA)
                    #cv2.resize(src, dsize=(640, 480), interpolation=cv2.INTER_AREA)

            # if self.preprocess :
                ###############
                # preprocess here
                ###############

                #
                # clip image range
                #
                # source_slice = np.where(source <= self.inputA_min, self.inputA_min, source)
                # source_slice = np.where(source_slice >= self.inputA_max, self.inputA_max, source)
                #
                # target_slice = np.where(target <= self.inputB_min, self.inputB_min, target)
                # target_slice = np.where(target_slice >= self.inputB_max, self.inputB_max, target)

                #
                # Z-score norm
                #
                # source_mean, source_std = np.mean(source_slice), np.std(source_slice)
                # source_slice = ((source_slice - source_mean) / source_std)
            source_slice = source_slice - np.min(source_slice)
            source_slice = source_slice / (np.max(source_slice) - np.min(source_slice))

            #
            # the intensity is scaled range of [0, 1]
            # #
            # source_volume = source_volume - self.inputA_min
            # source_volume = source_volume / (self.inputA_max - self.inputA_min)
            #
            #
            # target_volume = target_volume - self.inputB_min
            # target_volume = target_volume / (self.inputB_max - self.inputB_min)

            #
            # the intensity is scaled range of [-1, 1]
            #
            # if self.tanh_norm:
            #     source_volume = source_volume * 2 - 1
            #     target_volume = target_volume * 2 - 1

            if self.inputA_channel == 1:
                source_slice = np.array(source_slice[:, :,  np.newaxis])
            if self.inputB_channel == 1:
                target_slice = np.array(target_slice[:, :,  np.newaxis])

            source_slices.append(source_slice)
            target_slices.append(target_slice)

        return (np.array(source_slices), np.array(target_slices))

    def load_testset(self, listA,listB):

        source_slices = []
        target_slices = []

        for idx in range(len(listA)):
            source_slice = self.read_png(listA[idx])
            target_slice = self.read_npy(listB[idx])

            # padding image
            # if source_volume.shape[0] < self.inputA_size[0] or source_volume.shape[1] < self.inputA_size[1] or \
            #         source_volume.shape[2] < self.inputA_size[2]:
            #     source_volume = self.padding_volume(source_volume, self.inputA_min, self.inputA_size)
            #
            # if target_volume.shape[0] < self.inputB_size[0] or target_volume.shape[1] < self.inputB_size[1] or \
            #         target_volume.shape[2] < self.inputB_size[2]:
            #     target_volume = self.padding_volume(target_volume, self.inputB_min, self.inputB_size)

            source_slice = cv2.resize(source_slice, (self.inputA_size[0], self.inputA_size[1]), interpolation=cv2.INTER_AREA)
            target_slice = cv2.resize(target_slice, (self.inputB_size[0], self.inputB_size[1]),  interpolation=cv2.INTER_AREA)
            # if self.preprocess :
                ###############
                # preprocess here
                ###############

                #
                # clip image range
                #
                # source_slice = np.where(source <= self.inputA_min, self.inputA_min, source)
                # source_slice = np.where(source >= self.inputA_max, self.inputA_max, source)
                #
                # target_slice = np.where(target <= self.inputB_min, self.inputB_min, target)
                # target_slice = np.where(target >= self.inputB_max, self.inputB_max, target)

                #
                # Z-score norm
                #
                # source_mean, source_std = np.mean(source_slice), np.std(source_slice)
                # source_slice = ((source_slice - source_mean) / source_std)
            source_slice = source_slice - np.min(source_slice)
            source_slice = source_slice / (np.max(source_slice) - np.min(source_slice))


            #
            # the intensity is scaled range of [0, 1]
            # #
            # source_volume = source_volume - self.inputA_min
            # source_volume = source_volume / (self.inputA_max - self.inputA_min)
            #
            #
            # target_volume = target_volume - self.inputB_min
            # target_volume = target_volume / (self.inputB_max - self.inputB_min)

            #
            # the intensity is scaled range of [-1, 1]
            #
            # if self.tanh_norm:
            #     source_volume = source_volume * 2 - 1
            #     target_volume = target_volume * 2 - 1

            if self.inputA_channel == 1:
                source_slice = np.array(source_slice[:, :,  np.newaxis])
            if self.inputB_channel == 1:
                target_slice = np.array(target_slice[:, :,  np.newaxis])

            source_slices.append(source_slice)
            target_slices.append(target_slice)

        return np.array(source_slices), np.array(target_slices)

    def read_nii(self, inputImageFileName):

        reader = sitk.ImageFileReader()
        reader.SetImageIO("NiftiImageIO")
        reader.SetFileName(inputImageFileName)
        image = reader.Execute()

        numpyImage = sitk.GetArrayFromImage(image)
        slices = image.GetDepth()

        return (numpyImage, slices, image)

    def read_png(self, inputImageFileName):
        return cv2.imread(inputImageFileName)[:,:,0]
    def read_npy(self, inputImageFileName):
        return np.load(inputImageFileName)