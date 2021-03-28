import json
import numpy as np
import cdflib
import cv2 as cv
import matplotlib.pyplot as plt
import os

class SubjectReader():

    crop_shift = np.array([256, 256]) #We want 512x512 crop then resize down to 256x256.
    index_of_body_center = 11

    kpt_indices = np.array([1,2,3,6,7,8,14,24,25,26,28,17,18,19])

    data_folder = 'Data/'
    label_folder = 'Labels/'

    camera_idx_to_name = {
        0: '54138969',
        1: '55011271',
        2: '58860488',
        3: '60457274'
    }

    __file_open = False

    def __init__(self, subject, cam_indices = [0, 1, 2, 3]):
        self.subject = subject
        self.cam_indices = cam_indices

        #get the projection matrices
        with open('Labels/camera-parameters.json') as f:
            data = json.load(f)

        #Read the camera matrices
        R = []
        T = []
        CM = []
        for ind in cam_indices:
            _r = data['extrinsics'][subject][self.camera_idx_to_name[ind]]['R']
            _t = data['extrinsics'][subject][self.camera_idx_to_name[ind]]['t']
            _cm = data['intrinsics'][self.camera_idx_to_name[ind]]['calibration_matrix']

            R.append(_r)
            T.append(_t)
            CM.append(_cm)

        self.R = np.array(R)
        self.T = np.array(T)
        self.CM = np.array(CM)

        self._get_subject_data_names()
        self._open_new_file()


    def PrepData(self, action_name):

        vid_prefix = self.vid_prefix + action_name + '.'

        #read the data
        self.vids = []
        for ind in self.cam_indices:
            filename = vid_prefix + self.camera_idx_to_name[ind] + '.mp4'
            if os.path.isfile(filename) == False:
                print('video file not found: ', filename)
                quit()
            else:
                print('opening video file: ', filename)
            _v = cv.VideoCapture(filename)
            self.vids.append(_v)

        #read the 3d coords
        dat_prefix = self.dat_3d_prefix + action_name
        filename = dat_prefix + '.cdf'
        if os.path.isfile(filename) == False:
            print('video file not found: ', filename)
            quit()
        else:
            print('opening 3d data file: ', filename)
        self.X_3d = cdflib.CDF(filename).varget(0)[0]

        #read 2d uv coords
        uv_2d = []
        for ind in self.cam_indices:
            filename = self.dat_2d_prefix +  action_name + '.' + self.camera_idx_to_name[ind] + '.cdf'
            if os.path.isfile(filename) == False:
                print('video file not found: ', filename)
                quit()
            else:
                print('opening 2d data file: ', filename)
            _uvs = cdflib.CDF(filename).varget(0)[0]
            uv_2d.append(_uvs)

        self.uv_2d = np.array(uv_2d)
        self.frame_i = 0
        self.__file_open == True


    #this function is called once to check how many data are available per subject.
    def _get_subject_data_names(self):

        vid_prefix = self.data_folder + 'Videos_' + self.subject + '/' + self.subject + '/Videos/'
        data_files = os.listdir(vid_prefix)

        #get only the filenames
        data_files = [fn[:fn.find('.')] for fn in data_files]

        #get uniques
        data_files = list(set(data_files))

        data_motions = []
        #unique motions only
        for filename in data_files:
            if 'ALL' in filename: continue
            data_motions.append(filename)

        self.vid_prefix = vid_prefix
        self.dat_3d_prefix = self.label_folder + 'Poses_D3_Positions_' + self.subject + '/' + self.subject + '/MyPoseFeatures/D3_Positions/'
        self.dat_2d_prefix = self.label_folder + 'Poses_D2_Positions_' + self.subject + '/' + self.subject + '/MyPoseFeatures/D2_Positions/'
        self.motions = data_motions

    def _open_new_file(self):

        #on first call, we let this function shuffle the data files
        if self.__file_open == False:
            self.file_index = 0 #tracks which file is open

            #shuffle the list of actions
            np.random.shuffle(self.motions)

        #close video readers
        if self.__file_open:
            for vid in self.vids:
                vid.release()
            print('opening new file')

        #check if all files have been used. If yes, then reset counter and shuffle files list again
        if self.file_index == len(self.motions):
            self.file_index = 0
            np.random.shuffle(self.motions)

        self.PrepData(self.motions[self.file_index])
        #self.PrepData('Photo')

        #advance file index by one
        self.file_index += 1

    #this code will try to read frame data.
    #if end of video is reached, then it will tell the class to open new file and return False.
    def _get_frame_data(self, skip_i):

        #skip frames
        for _ in range(skip_i):
            for cam_id in self.cam_indices:
                ret, _ = self.vids[cam_id].read()
            if ret == False:
                self._open_new_file()
                return None, None, None, None, None, False
            self.frame_i += 1

        frames = []
        crops = []
        X = []
        uvs = []
        Ps = []
        heatmaps = []

        #read frames
        for cam_id in self.cam_indices:
            ret, _frame = self.vids[cam_id].read()
            if ret == False:
                return None, None, None, None, None, False

            frames.append(_frame[:1000,:1000])
        self.frame_i += 1

        if self.frame_i >= len(self.X_3d):
            return None,None,None,None,None,False

        #get corresponding 3d pos
        X = self.X_3d[self.frame_i]
        X = X.reshape((-1, 3))
        X = X[self.kpt_indices]
        S = np.ones((X.shape[0],1))
        X = np.concatenate([X, S], axis = -1)
        X = X.reshape((X.shape[0], 4, 1))

        #2d points are projected on the fly for each camera
        for cam_id in self.cam_indices:

            #get camera matrices
            _r = self.R[cam_id]
            _t = self.T[cam_id]
            _cm = self.CM[cam_id]
            _rt = np.concatenate([_r, _t], axis = -1)

            #get the center of the crop and top left coords
            _uvs = self.uv_2d[cam_id][self.frame_i]
            _uvs = _uvs.reshape((-1, 2))
            _crop_center = np.round(_uvs[self.index_of_body_center]).astype('int32')
            _crop_coord = (_crop_center - self.crop_shift)

            #add random shifting
            _crop_coord += np.random.normal(0, 20, (2)).astype('int32')

            #ensure crop stays within the frame
            if _crop_coord[1] < 0: _crop_coord[1] = 0
            if _crop_coord[1] + 2 * self.crop_shift[1] >= frames[cam_id].shape[1]:
                _crop_coord[1] += frames[cam_id].shape[1] - (_crop_coord[1] + 2 * self.crop_shift[1])

            if _crop_coord[0] < 0: _crop_coord[0] = 0
            if _crop_coord[0] + 2 * self.crop_shift[0] >= frames[cam_id].shape[0]:
                _crop_coord[0] += frames[cam_id].shape[0] - (_crop_coord[0] + 2 * self.crop_shift[0])

            #get the crop from the frame
            _crop = frames[cam_id][_crop_coord[1]:_crop_coord[1] + 2 * self.crop_shift[1],
                                   _crop_coord[0]:_crop_coord[0] + 2 * self.crop_shift[0]]

            #resise the crop to 256x256
            _crop = cv.resize(_crop, (256,256))

            #need to shift the center of the camera for the crop
            crop_matrix = np.array([[0,   0,    _crop_coord[0]],
                                    [0,   0,    _crop_coord[1]],
                                    [0,   0,                0]])

            #modify intrinsic matrix for crop
            _cm_crop = _cm - crop_matrix
            _P = _cm_crop @ _rt

            #modify projection matrix for resizing
            n = -1# resize power
            resize_mat = np.array([[2**n,     0,   2**(n - 1) - 0.5],
                                   [0,     2**n,   2**(n - 1) - 0.5],
                                   [0,        0,                  1]])
            _P = resize_mat @ _P
            _uv = _P @ X
            _uv = _uv[:,:,0]
            _uv = _uv/np.reshape(_uv[:,2], (_uv.shape[0], 1))
            _uv = _uv[:,:2]

            #check all uvs are withing the crop
            for kpt in _uv:
                if kpt[0] > 256 or kpt[1] > 256:
                    print('some kpts outside frame')
                    return None, None, None, None, None, False

            #create heatmaps
            _hms = np.zeros((_uv.shape[0], 64, 64))
            for i, _kp in enumerate(_uv):
                _kp = np.round(_kp)
                if _kp[0] < 256 and _kp[1] < 256:
                    _hms[i][int(_kp[1]//4), int(_kp[0]//4)] = 1

                    # plt.imshow(cv.resize(_crop, (64,64))[:,:,[2,1,0]])
                    # plt.imshow(_hms[i], alpha = 0.3)
                    # plt.show()

            uvs.append(_uv)
            Ps.append(_P)
            crops.append(_crop)
            heatmaps.append(_hms)

        crops = np.array(crops)
        uvs = np.array(uvs)
        Ps = np.array(Ps)
        heatmaps = np.array(heatmaps); heatmaps = np.transpose(heatmaps, [0,2,3,1])
        print(int(self.vids[0].get(cv.CAP_PROP_POS_FRAMES)))
        print(self.frame_i)

        return crops, X, uvs, Ps, heatmaps, True


    def GetNext(self, count = 4, skip_frame = 30, skip_range = 15):

        frames = []
        Xs = []
        uvs = []
        Ps = []
        heatmaps = []

        for f in range(count):
            while(True):
                frames_to_skip = np.random.uniform(skip_frame - skip_range, skip_frame + skip_range)
                frames_to_skip = int(np.rint(frames_to_skip))
                _crops, _xs, _uvs, _ps, _heatms, ret = self._get_frame_data(frames_to_skip)
                if ret: break

            # print(_crops.shape)
            # print(_xs.shape)
            # print(_ps.shape)
            # print(_uvs.shape)
            #
            # for crop, P, f_uv in zip(_crops, _ps, _uvs):
            #     plt.imshow(crop[:,:,[2,1,0]])
            #
            #     proj = P @ _xs
            #     proj = proj[:,:,0]
            #     proj = proj/np.reshape(proj[:,2], (proj.shape[0], 1))
            #     proj = proj[:,:2]
            #
            #     plt.scatter(proj[:,0], proj[:,1], c = 'white', s = 2)
            #     plt.scatter(f_uv[:,0], f_uv[:,1] + 1, c = 'red', s = 2)
            #
            #     plt.show()

            frames.append(_crops)
            Xs.append(_xs)
            uvs.append(_uvs)
            Ps.append(_ps)
            heatmaps.append(_heatms)

        frames = np.array(frames)
        Xs = np.array(Xs)
        uvs = np.array(uvs)
        Ps = np.array(Ps)
        heatmaps = np.array(heatmaps)

        # print(frames.shape)
        # print(Xs.shape)
        # print(uvs.shape)
        # print(Ps.shape)
        # print(heatmaps.shape)

        return frames, Xs, uvs, Ps, heatmaps

if __name__ == '__main__':

    reader = SubjectReader('S11')
    frames, Xs, uvs, Ps, heatmaps = reader.GetNext()
    print(frames.shape)
