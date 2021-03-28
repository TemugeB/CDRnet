import tensorflow as tf
from SubjectReader import SubjectReader
from model_define import GetModel, SII, FTL_inv, FTL
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

"""
inputs:
reader - SubjectReader class defined in SubjectReader.py. For reading data.
count - Number of frames to pull from each video.
"""
def GetData(readers, count = 10):

    #camera frames container
    frames_v1 = []
    frames_v2 = []
    frames_v3 = []
    frames_v4 = []

    #projection matrices container
    pms_v1 = []
    pms_v2 = []
    pms_v3 = []
    pms_v4 = []

    #inverse(pseudo) projection matrix container
    pinvs_v1 = []
    pinvs_v2 = []
    pinvs_v3 = []
    pinvs_v4 = []

    #keypoint heatmaps.
    heatmaps_v1 = []
    heatmaps_v2 = []
    heatmaps_v3 = []
    heatmaps_v4 = []

    #xy coords of keypoints in each frame
    uvs_v1 = []
    uvs_v2 = []
    uvs_v3 = []
    uvs_v4 = []

    #3D coords in world space
    Xs = []

    """
    a reader class is genereated for each actor in the videos.
    We call the GetNext() method to get some sample frames and corresponding labels.
    """
    for reader in readers:
        #frames : [batch_size, 4 views, 256, 256, 3], camera views.
        #Xs: [batch_size, 14 keypoints, 4 xyzw coords, 1], 3D coords in homogeneous coords
        #uvs: [batch_size, 4 views, 14 keypoints, 2 uv coords] keypoint coords in each image
        #Ps: [batch_size, 4 views, 3, 4] #projection matrices
        #heatmaps: [batch_size, 4, 64, 64, 14] #heatmap of each keypoint. Not used for training in this implementation
        frames, _xs, uvs, Ps, heatmaps = reader.GetNext(count)

        Xs.append(_xs)

        frames_v1.append(frames[:,0])
        frames_v2.append(frames[:,1])
        frames_v3.append(frames[:,2])
        frames_v4.append(frames[:,3])

        pms_v1.append(Ps[:,0])
        pms_v2.append(Ps[:,1])
        pms_v3.append(Ps[:,2])
        pms_v4.append(Ps[:,3])

        heatmaps_v1.append(heatmaps[:,0])
        heatmaps_v2.append(heatmaps[:,1])
        heatmaps_v3.append(heatmaps[:,2])
        heatmaps_v4.append(heatmaps[:,3])

        uvs_v1.append(uvs[:,0])
        uvs_v2.append(uvs[:,1])
        uvs_v3.append(uvs[:,2])
        uvs_v4.append(uvs[:,3])


    Xs = np.array(Xs).reshape((-1, 14, 4, 1))

    frames_v1 = np.array(frames_v1).reshape((-1, 256, 256, 3))
    frames_v2 = np.array(frames_v2).reshape((-1, 256, 256, 3))
    frames_v3 = np.array(frames_v3).reshape((-1, 256, 256, 3))
    frames_v4 = np.array(frames_v4).reshape((-1, 256, 256, 3))

    pms_v1 = np.array(pms_v1).reshape((-1, 3, 4))
    pms_v2 = np.array(pms_v2).reshape((-1, 3, 4))
    pms_v3 = np.array(pms_v3).reshape((-1, 3, 4))
    pms_v4 = np.array(pms_v4).reshape((-1, 3, 4))

    heatmaps_v1 = np.array(heatmaps_v1).reshape((-1, 64, 64, 14))
    heatmaps_v2 = np.array(heatmaps_v2).reshape((-1, 64, 64, 14))
    heatmaps_v3 = np.array(heatmaps_v3).reshape((-1, 64, 64, 14))
    heatmaps_v4 = np.array(heatmaps_v4).reshape((-1, 64, 64, 14))

    uvs_v1 = np.array(uvs_v1).reshape((-1, 14, 2))
    uvs_v2 = np.array(uvs_v2).reshape((-1, 14, 2))
    uvs_v3 = np.array(uvs_v3).reshape((-1, 14, 2))
    uvs_v4 = np.array(uvs_v4).reshape((-1, 14, 2))


    #modify P matrix for resizing
    #the heatmaps are size 64x64.
    #So we need to downsize the 256x256 frames by 2^2, which means the projection matrix also needs to be modified.
    n = -2# resize power
    resize_mat = np.array([[2**n,     0,   2**(n - 1) - 0.5],
                          [0,     2**n,   2**(n - 1) - 0.5],
                          [0,        0,                  1]])

    resize_mat = resize_mat.reshape((1, 3, 3))
    pms_v1 = resize_mat @ pms_v1
    pms_v2 = resize_mat @ pms_v2
    pms_v3 = resize_mat @ pms_v3
    pms_v4 = resize_mat @ pms_v4

    #get the inverse matrices. The scale of the image here is 64x64, for which I'm getting the inverse projection matrix.
    for P in pms_v1: pinvs_v1.append(linalg.pinv(P))
    for P in pms_v2: pinvs_v2.append(linalg.pinv(P))
    for P in pms_v3: pinvs_v3.append(linalg.pinv(P))
    for P in pms_v4: pinvs_v4.append(linalg.pinv(P))
    pinvs_v1 = np.array(pinvs_v1)
    pinvs_v2 = np.array(pinvs_v2)
    pinvs_v3 = np.array(pinvs_v3)
    pinvs_v4 = np.array(pinvs_v4)


    print('view xs: ', Xs.shape)
    print('view frames: ', frames_v1.shape)
    print('view Ps: ', pms_v1.shape)
    print('view Pinvs: ', pinvs_v1.shape)
    print('view heatmaps: ', heatmaps_v1.shape)
    print('view uvs: ', uvs_v1.shape)

    frames = [frames_v1, frames_v2, frames_v3, frames_v4]
    pms = [pms_v1, pms_v2, pms_v3, pms_v4]
    pinvs = [pinvs_v1, pinvs_v2, pinvs_v3, pinvs_v4]
    heatmaps = [heatmaps_v1, heatmaps_v2, heatmaps_v3, heatmaps_v4]
    uvs = [uvs_v1, uvs_v2, uvs_v3, uvs_v4]

    return Xs, frames, pms, pinvs, heatmaps, uvs

"""
Use this loss if you want to add the differentiable direct linear transform layer to backpropagate through.
Sometimes the DLT layer does not get the 3D coords and returns NaN value.
This loss simply removes the NaN values and applies MSE loss on the remaining predictions.
"""
def _DLT_loss(y_true, y_pred):

    y_true = tf.reshape(y_true, (-1, 3))
    y_pred = tf.reshape(y_pred, (-1, 3))
    nans = tf.math.is_nan(y_pred[:,0])
    not_nans = tf.math.logical_not(nans)

    print(y_pred[nans].shape)

    if y_pred[not_nans].shape[0] != 0:
        return tf.reduce_mean(tf.keras.losses.mse(y_pred[not_nans], y_true[not_nans]))
    else:
        return 0

"""
This loss is added to explicitly turn off back propagation through the DLT layer.
If you want to backpropagate through the DLT layer, remove this loss and add the _DLT_loss in the model.compile() call.
"""
def _dummy_loss(y_true, y_pred):
    return 0 * y_pred


"""
Main training loop. My model was trained for 7500 epochs to converge. You can try around that number.
The training loss went down to about 2.5, which is about 0.6 MSE per camera view.
"""
def TrainModel(model, epochs = 7501):

    #Which subjects to train on.
    """
    # WARNING: S9 has multiple wrong labeling which will significantly hinder network  training.
    Make sure to remove those videos and labels
    """
    subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']

    #get the subject readers
    readers = []
    for subject in subjects:
        reader = SubjectReader(subject)
        readers.append(reader)

    loss = []
    for ep in range(epochs):
        Xs, frames, pms, pinvs, heatmaps, uvs = GetData(readers, 10)
        Xs = Xs.reshape((-1, 14, 4))
        Xs = Xs[:,:,:3]

        hist = model.fit([frames[0], frames[1], frames[2], frames[3],
                          pms[0], pms[1], pms[2], pms[3],
                          pinvs[0], pinvs[1], pinvs[2], pinvs[3]],
                          [uvs[0]/4, uvs[1]/4, uvs[2]/4, uvs[3]/4, Xs], batch_size = 7)
        loss.append(hist.history['loss'])

        plt.plot(loss)
        plt.savefig('loss.png')
        plt.clf()

        if ep%500 == 0 and ep != 0:
            model.save('models/canonical_fusion_ep' + str(ep).zfill(4)+ '.h5')


def train():

    """In the compile below, if you want to use differentiable DLT to backpropagate, then change _dummy_loss to _DLT_loss.
       However I don't recommend it when you start training. This will lead to massive instability and network will fail training.
       In the paper, the authors suggest to turn on the differentiable DLT at the very end. I tried this and still had some instabiliy.
       However, adding the differentiable DLT loss only increases the network accuracy by a little bit. So you're not missing out much if you remove it.
       The trained model I provide did not use the _DLT_loss.
    """
    CDRnet, encoder, decoder = GetModel()
    CDRnet.compile(optimizer="Adam", loss=["mse","mse","mse","mse", _dummy_loss])

    """Use the code below if you need to load saved model. Otherwise, expect errors"""
    #CDRnet = tf.keras.models.load_model('models/canonical_fusion_ep2000.h5', custom_objects={'SII': SII, '_dummy_loss':_dummy_loss, 'FTL_inv': FTL_inv, 'FTL': FTL})

    TrainModel(CDRnet)


def TestModel(model):

    subjects = ['S1']

    #get the subject readers
    readers = []
    for subject in subjects:
        reader = SubjectReader(subject)
        readers.append(reader)

    get_count = 3
    Xs, frames, pms, pinvs, heatmaps, uvs = GetData(readers, count = get_count)
    Xs = Xs.reshape((-1, 14, 4))
    Xs = Xs[:,:,:3]

    x_v1_kpts, x_v2_kpts, x_v3_kpts, x_v4_kpts, recons = model.predict([frames[0], frames[1], frames[2], frames[3],
                                                                        pms[0], pms[1], pms[2], pms[3],
                                                                        pinvs[0], pinvs[1], pinvs[2], pinvs[3]])

    kpts = np.stack([x_v1_kpts, x_v2_kpts, x_v3_kpts, x_v4_kpts], axis = 1)

    #keypoints to draw lines between to form skeleton
    connections = [[6,7], [7,8], [8,9], [9,10], [7,11], [11,12], [12, 13], [7,0], [0,1], [1,2], [7,3], [3,4], [4,5]]
    recons = np.array(recons) #3d reconstructed points

    #plot out predictions
    from mpl_toolkits.mplot3d import Axes3D
    for i in range(get_count):
        plt.figure(figsize = (8,8))

        #show the keypoints
        for view in range(4):
            plt.subplot(2,2,view + 1)
            plt.imshow(frames[view][i][:,:,[2,1,0]])
            for kpt in range(14):
                plt.scatter([uvs[view][i][kpt][0]], [uvs[view][i][kpt][1]], c = 'white', s = 2)
                plt.scatter([4 * kpts[i][view][kpt][0]], [4 * kpts[i][view][kpt][1]], c = 'red', s = 2)
        plt.legend()
        plt.show()
        plt.close()

        #show 3d reconstruction
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim3d(-700, 700)
        ax.set_ylim3d(-700, 700)
        ax.set_zlim3d(0, 1400)
        for _c in connections:
            ax.plot(xs = [Xs[i][_c[0],0], Xs[i][_c[1],0]], ys = [Xs[i][_c[0],1], Xs[i][_c[1],1]], zs = [Xs[i][_c[0],2], Xs[i][_c[1],2]], c = 'black')
            ax.plot(xs = [recons[i][_c[0],0], recons[i][_c[1],0]], ys = [recons[i][_c[0],1], recons[i][_c[1],1]], zs = [recons[i][_c[0],2], recons[i][_c[1],2]], c = 'red')

        ax.plot([], [], c = 'black', label = 'true')
        ax.plot([], [], c = 'red', label = 'prediction')

        plt.legend()
        plt.show()
        plt.close()

    pass


def test():

    """Use the code below if you need to load saved model. Otherwise, expect errors"""
    CDRnet = tf.keras.models.load_model('models/canonical_fusion_ep7500.h5', custom_objects={'SII': SII, '_dummy_loss':_dummy_loss, 'FTL_inv': FTL_inv, 'FTL': FTL})
    TestModel(CDRnet)


if __name__ == '__main__':

    #uncomment which you want here.
    #train()
    test()
    pass
