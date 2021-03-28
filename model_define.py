import tensorflow as tf

"""
Inverse feature transform layer.
Check the paper for details.
They don't specify what actually is the inverse of the projection matrix is.
Since projection matrix is 3x4, it is not invertible.
In this implementation, I use the pseudo-inverse of the projection matrix.
"""
class FTL_inv(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(FTL_inv, self).__init__(**kwargs)

    def get_config(self):
        config = super(FTL_inv, self).get_config()
        return config

    def call(self, inputs):

        #inverse projection matrix input is (batch_size, 4, 3)
        #latent vector input size is (batch_size, 8, 8, 300)
        P = inputs[0]
        z = inputs[1]

        batch_size = tf.shape(z)[0]

        z = tf.reshape(z, (batch_size, 8, 8, 100, 3, 1))
        P = tf.reshape(P, (batch_size, 1, 1, 1, 4, 3))
        inverted = P @ z

        return tf.reshape(inverted, (batch_size, 8, 8, 400))


"""
Feature transform layer.
Check the paper for details.
"""
class FTL(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(FTL, self).__init__(**kwargs)

    def get_config(self):
        config = super(FTL, self).get_config()
        return config

    def call(self, inputs):

        #projection matrix input is size (batch_size, 3, 4)
        #latent vector input size is (batch_size, 8, 8, 300)

        P = inputs[0]
        z = inputs[1]
        batch_size = tf.shape(z)[0]

        #need to add homogeneous coordinate to last layer
        z = tf.reshape(z, (batch_size, 8, 8, 100, 4, 1))
        P = tf.reshape(P, (batch_size, 1, 1, 1, 3, 4))
        projected = P @ z
        return tf.reshape(projected, (batch_size, 8, 8, 300))


"""
Differentiable direct linear transform layer in the paper.
In this implementation I call it Shifted Inverse Iterations(SII), which is the algorithm it is based on.
The original implementation was written in pytorch by the authors of the paper.
Here I simply translated from pytorch to tensorflow function calls.
"""
class SII(tf.keras.layers.Layer):
    def __init__(self, num_iters = 5, **kwargs):
        super(SII, self).__init__(**kwargs)
        self.num_iters = num_iters

    def get_config(self):
        config = super(SII, self).get_config()
        config.update({"num_iters": self.num_iters})
        return config

    def homogeneous_to_euclidean(self,v):
        # WARNING: This function implicitly convets tf.Variable to tf.tensor.
        if len(v.shape) > 1:
            return tf.transpose(tf.transpose(v[:,:-1])/v[:,-1])
        else:
            return v[:-1]/v[-1]

    def call(self, input):

        M = input[0]
        pts = input[1]

        batch_size = tf.shape(M)[0]
        views = M.shape[1]

        #We need to solve Ax = 0. In this case, A = (u x P) for a single camera, with u = camera pixel coords, P = camera projection matrix
        #However, only A[0:2] is needed, since the last row is a linear combination of the first 2.
        A = tf.repeat(M[:,:,2:3], repeats = 2, axis = 2) * tf.reshape(pts, (batch_size, pts.shape[1], 2, 1)) - M[:,:,0:2]
        A = tf.reshape(A, (batch_size, 2 * views, 4))

        #contruct (A - alpha * I) for SII
        alpha = tf.constant(0.001)
        B = tf.transpose(A, perm = [0,2,1]) @ A - tf.repeat(tf.reshape(alpha * tf.eye(4), (1,4,4)), repeats = batch_size, axis = 0)

        #initial guess for triangulated point is randomly generated.
        #A good starting point is near [0,0,0,1], assuming the world coords are setup to be (0,0,0) at the center of the real camera space
        X = tf.random.normal((batch_size, 4, 1), mean = 0.5, stddev = 0.5 , dtype = 'float32')

        #solve By = X for y. From SII, X = y/|y|
        for _ in range(self.num_iters):
            X =  tf.linalg.solve(B, X)
            X = X/tf.expand_dims(tf.norm(X, axis = 1), axis = -1)

        X = tf.reshape(X, (batch_size, 4))

        return self.homogeneous_to_euclidean(X)


"""
Decoder model that inputs camera disentangled representation and outputs heatmaps.
"""
def decoder_model(in_shape):

    in_feats = tf.keras.layers.Input(in_shape)

    #8x8x2048 to 16x16x256
    x = tf.keras.layers.Conv2DTranspose(256, 1, 2)(in_feats)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)

    #16x16x256 to 32x32x256
    x = tf.keras.layers.Conv2DTranspose(256, 1, 2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)

    #32x32x256 to 64x64x256
    x = tf.keras.layers.Conv2DTranspose(256, 1, 2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)

    #64x64x256 to 64x64x14
    out_heatmaps = tf.keras.layers.Conv2D(14, 1, activation = 'sigmoid')(x)

    model = tf.keras.models.Model(in_feats, out_heatmaps, name = 'decoder')

    return model


"""
This calculates the center of mass of each heatmap to determine keypoint.
Don't use this if you have more than 1 sunject in the frame.
"""
def _calculate_heatmap_keypoints(linspace, heatmaps):

    h_y = tf.reduce_sum(linspace * tf.reduce_sum(heatmaps, axis = -2), axis = -2)/tf.reduce_sum(heatmaps, axis = [-3, -2])
    h_x = tf.reduce_sum(linspace * tf.reduce_sum(heatmaps, axis = -3), axis = -2)/tf.reduce_sum(heatmaps, axis = [-3, -2])

    return tf.transpose(tf.stack([h_x - 1, h_y - 1], axis = 1), [0, 2, 1])


"""
Main model code.
I've decided to keep each camera view as separate network input and output.
This is done because tensor reshape operations can get confusing so this makes a more readable code.
However, the code is hard-coded to support 4 camera views.
But changing to 2 or 3 views should be relatively simple.
"""
def GetModel(heatmap_size = 64):

    resnet = tf.keras.applications.ResNet152(weights = 'imagenet', input_shape = (256,256,3), include_top = False)
    encoder = tf.keras.models.Model(inputs = resnet.input, outputs = resnet.layers[514].output, name = 'encoder')
    decoder = decoder_model((8,8,2048))
    sii = SII(2, name = 'SII')
    ftl_inv = FTL_inv(name = 'FTL_inv')
    ftl = FTL(name = 'FTL')

    #for center of mass calculations of heatmaps
    linspace = tf.range(1, heatmap_size + 1, dtype = 'float32')
    linspace = tf.reshape(linspace, (64, 1))

    #input scene views
    in_img_v1 = tf.keras.layers.Input((256,256,3))
    in_img_v2 = tf.keras.layers.Input((256,256,3))
    in_img_v3 = tf.keras.layers.Input((256,256,3))
    in_img_v4 = tf.keras.layers.Input((256,256,3))

    #input projection matrices
    pm_v1_input = tf.keras.layers.Input((3,4))
    pm_v2_input = tf.keras.layers.Input((3,4))
    pm_v3_input = tf.keras.layers.Input((3,4))
    pm_v4_input = tf.keras.layers.Input((3,4))


    #input inverse projection matrices(pseudo inverses)
    pm_inv_v1_input = tf.keras.layers.Input((4,3))
    pm_inv_v2_input = tf.keras.layers.Input((4,3))
    pm_inv_v3_input = tf.keras.layers.Input((4,3))
    pm_inv_v4_input = tf.keras.layers.Input((4,3))


    #encoded scene views
    x_v1 = encoder(in_img_v1)
    x_v2 = encoder(in_img_v2)
    x_v3 = encoder(in_img_v3)
    x_v4 = encoder(in_img_v4)

    #combine the scene views into a single tensor
    #first reduce dims to 300 feature maps per view.
    shared_conv_layer = tf.keras.layers.Conv2D(300, 1, padding = 'same', name = 'shared_2048_to_300')

    x_v1 = shared_conv_layer(x_v1)
    x_v1 = tf.keras.layers.BatchNormalization()(x_v1)
    x_v1 = tf.keras.activations.relu(x_v1)

    x_v2 = shared_conv_layer(x_v2)
    x_v2 = tf.keras.layers.BatchNormalization()(x_v2)
    x_v2 = tf.keras.activations.relu(x_v2)

    x_v3 = shared_conv_layer(x_v3)
    x_v3 = tf.keras.layers.BatchNormalization()(x_v3)
    x_v3 = tf.keras.activations.relu(x_v3)

    x_v4 = shared_conv_layer(x_v4)
    x_v4 = tf.keras.layers.BatchNormalization()(x_v4)
    x_v4 = tf.keras.activations.relu(x_v4)

    #FTL inverse applied here
    x_v1 = ftl_inv([pm_inv_v1_input, x_v1])
    x_v2 = ftl_inv([pm_inv_v2_input, x_v2])
    x_v3 = ftl_inv([pm_inv_v3_input, x_v3])
    x_v4 = ftl_inv([pm_inv_v4_input, x_v4])

    #concatenate into single feature map over all views
    can_fusion = tf.concat([x_v1, x_v2, x_v3, x_v4], axis = -1)
    #fusion of views
    can_fusion = tf.keras.layers.Conv2D(400, 1, padding = 'same')(can_fusion)
    can_fusion = tf.keras.layers.BatchNormalization()(can_fusion)
    can_fusion = tf.keras.activations.relu(can_fusion)
    #second 1x1 convolusion
    can_fusion = tf.keras.layers.Conv2D(400, 1, padding = 'same')(can_fusion)
    can_fusion = tf.keras.layers.BatchNormalization()(can_fusion)
    can_fusion = tf.keras.activations.relu(can_fusion)

    #project to separate views.
    x_v1 = ftl([pm_v1_input, can_fusion])
    x_v2 = ftl([pm_v2_input, can_fusion])
    x_v3 = ftl([pm_v3_input, can_fusion])
    x_v4 = ftl([pm_v4_input, can_fusion])

    #lift back to 2048 channels
    shared_conv_layer2 = tf.keras.layers.Conv2D(2048, 1, padding = 'same', name = 'shared_300_to_2048')
    x_v1 = shared_conv_layer2(x_v1)
    x_v1 = tf.keras.layers.BatchNormalization()(x_v1)
    x_v1 = tf.keras.activations.relu(x_v1)

    x_v2 = shared_conv_layer2(x_v2)
    x_v2 = tf.keras.layers.BatchNormalization()(x_v2)
    x_v2 = tf.keras.activations.relu(x_v2)

    x_v3 = shared_conv_layer2(x_v3)
    x_v3 = tf.keras.layers.BatchNormalization()(x_v3)
    x_v3 = tf.keras.activations.relu(x_v3)

    x_v4 = shared_conv_layer2(x_v4)
    x_v4 = tf.keras.layers.BatchNormalization()(x_v4)
    x_v4 = tf.keras.activations.relu(x_v4)

    #decode view specific encodings. Input: [batch, 8, 8, 2048], output: [batch, 64, 64, 14]
    x_v1_heatmap = decoder(x_v1)
    x_v2_heatmap = decoder(x_v2)
    x_v3_heatmap = decoder(x_v3)
    x_v4_heatmap = decoder(x_v4)

    #Get center of mass of heatmaps. Output is [batch, 14, 2] per view
    x_v1_kpts_out = _calculate_heatmap_keypoints(linspace, x_v1_heatmap)
    x_v2_kpts_out = _calculate_heatmap_keypoints(linspace, x_v2_heatmap)
    x_v3_kpts_out = _calculate_heatmap_keypoints(linspace, x_v3_heatmap)
    x_v4_kpts_out = _calculate_heatmap_keypoints(linspace, x_v4_heatmap)

    #Input to DLT and get 3D coords
    #first [batch, 14, 2] to [batch, 14, 1, 2]. Then concat along views and get [batch, 14, n_views, 2]
    x_v1_kpts = tf.reshape(x_v1_kpts_out, (-1, 14, 1, 2))
    x_v2_kpts = tf.reshape(x_v2_kpts_out, (-1, 14, 1, 2))
    x_v3_kpts = tf.reshape(x_v3_kpts_out, (-1, 14, 1, 2))
    x_v4_kpts = tf.reshape(x_v4_kpts_out, (-1, 14, 1, 2))
    view_collected = tf.concat([x_v1_kpts, x_v2_kpts, x_v3_kpts, x_v4_kpts], axis = 2)
    #convert to set of view paired points as: [batch * num_keypoints, num_views, 2]
    view_collected = tf.reshape(view_collected, [-1, 4, 2])

    #prepare the projection matrices. One projection matrix is input per frame
    #first repeat projection matrix for all keypoint heatmaps
    pm_v1 = tf.reshape(pm_v1_input, (-1, 1, 3, 4))
    pm_v1 = tf.repeat(pm_v1, 14, axis = 1)
    pm_v2 = tf.reshape(pm_v2_input, (-1, 1, 3, 4))
    pm_v2 = tf.repeat(pm_v2, 14, axis = 1)
    pm_v3 = tf.reshape(pm_v3_input, (-1, 1, 3, 4))
    pm_v3 = tf.repeat(pm_v3, 14, axis = 1)
    pm_v4 = tf.reshape(pm_v4_input, (-1, 1, 3, 4))
    pm_v4 = tf.repeat(pm_v4, 14, axis = 1)

    #concatenate along all view points
    pm_v1 = tf.reshape(pm_v1, (-1, 14, 1, 3, 4))
    pm_v2 = tf.reshape(pm_v2, (-1, 14, 1, 3, 4))
    pm_v3 = tf.reshape(pm_v3, (-1, 14, 1, 3, 4))
    pm_v4 = tf.reshape(pm_v4, (-1, 14, 1, 3, 4))
    P_collected = tf.concat([pm_v1, pm_v2, pm_v3, pm_v4], axis = 2)
    P_collected = tf.reshape(P_collected, [-1, 4, 3, 4])

    #the separate views are being sent to DLT in this function call.
    recons = sii([P_collected, view_collected])
    recons = tf.reshape(recons, (-1, 14, 3))

    """
    Explanation of network inputs and outputs.
    Inputs:
    4 x Camera views: (batch_size, 256, 256, 3)
    4 x Projection matrices: (batch_size, 3, 4)
    4 x Inverse(Pseudo) projection matrices: (batch_size, 4, 3)
    Outputs:
    4 x Detected keypoints: (batch_size, 14, 2), where 14 is the number of keypoints and 2 is the x,y coords in image space
    1 x 3D reconstruction. (batch_size, 14, 3), where 14 is the number of keypoints and 3 is the x,y,z coords in world space.
    """
    CDRnet = tf.keras.models.Model(inputs = [in_img_v1, in_img_v2, in_img_v3, in_img_v4,
                                             pm_v1_input, pm_v2_input, pm_v3_input, pm_v4_input,
                                             pm_inv_v1_input, pm_inv_v2_input, pm_inv_v3_input, pm_inv_v4_input],
                                  outputs = [x_v1_kpts_out, x_v2_kpts_out, x_v3_kpts_out, x_v4_kpts_out, recons])
    CDRnet.summary()

    return CDRnet, encoder, decoder

if __name__ == '__main__':

    #Encoder is the ResNet152 as used in the paper.
    #decoder is the output heatmaps. If you want the heatmaps only, then you can use the decoder only.
    CDRnet, encoder, decoder = GetModel()
