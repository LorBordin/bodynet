from tensorflow.python.ops.image_ops import _ImageDimensions
import tensorflow as tf

@tf.function
def create_density_maps(peak_coords: tf.Tensor, grid_dim: int):
    """ 
        Returns several density maps of size (grid_dim, grid_dim), peaked in 
        cell corresponding to peak_coords.The number of mapsis equal to  the 
        number of peak_coords.

        Parameters
        ----------
        peak_coords: tf.Tensor 
            Tensor of shape (n_joints, 2) containing the normalised coordinates
            of the peaks.
        grid_dim: int
            Side length of the density maps. It is equal to the FPN output.
        
        Returns
        -------
        pdfs: tf.Tensor
            Density maps of shape (grid_dim ** 2, n_joints).

    """
    # build the map
    grid = tf.cast(tf.linspace(0, 1, grid_dim+1), tf.float32)
    grid += 1/(grid_dim)/2
    grid = grid[:-1]
    xx, yy = tf.meshgrid(grid, grid)
    
    # define a map for each pair of coordinates
    n_joints = len(peak_coords)
    XX = tf.tile(tf.expand_dims(xx,-1), [1, 1, n_joints])
    YY = tf.tile(tf.expand_dims(yy,-1), [1, 1, n_joints])
    
    # define the PDFs
    X, Y = peak_coords[:,0], peak_coords[:,1]
    sigma = 1.0 / 2 / grid_dim
    pdfs = - (tf.math.pow(XX-X, 2) + tf.math.pow(YY-Y, 2)) \
        / (2*tf.math.pow(sigma, 2))
    pdfs = tf.math.exp(pdfs)
    pdfs = pdfs / tf.math.reduce_max(pdfs, axis=[0,1])
    pdfs = tf.where(tf.math.is_nan(pdfs), 0.0, pdfs)
    
    # deal with coords zero coords (not visible): set the PDFs to zero 
    mask = tf.where(tf.math.reduce_any(peak_coords==0., axis=-1), 0., 1.)
    mask = tf.reshape(mask, (1, 1, n_joints))

    pdfs *= mask
    pdfs = tf.reshape(pdfs, (grid_dim*grid_dim, -1))
    
    return pdfs


def sum_density_maps(pdfs):
    """ 
        Sums together multiple density maps in a whistful way.

        Parameters
        ----------
        pdfs: tf.Tensor
            Density maps of shape (grid_dim ** 2, n_joints).

        Returns
        -------
        pdf: tf.Tensor
            Multi peaked density map of shape (grid_dim ** 2, 1).
    """
    pdf = tf.math.reduce_sum(pdfs, axis=-1)
    pdf = tf.where(pdf>1.0, 1.0, pdf)
    return tf.expand_dims(pdf, axis=-1)


def crop_roi(
        img, 
        c_kpts, 
        probas, 
        centres,
        labels, 
        use_random_margin = False,
        min_margin = 0.01,
        mean_margin = .15, 
        confidence_thres = .05,
    ):
    """ 
    Crops the image focussing on the Region Of Interest (ROI). To get the 
    ROI it uses the keypoints coordinates (labels if during training or 
    previous predictions if in production).
    
    Assume coords are in the shape (-1, 3) and img in format [H, W, C]

    Parameters
    ----------
    img : tf.tensor[float]
        ...
    c_kpts : tf.tensor[float]
        ...
    probas : tf.tensor[float]
        ...
    labels : ...
        ...
    use_random_margin : bool
        ...
    min_margin : float
        ...
    max_margin : float
        ...
    confidence_thres : float

    Returns
    -------
    img_crop : ...
        ...
    new_labels : ...
        ...
    
    """ 
    # define margin 
    if use_random_margin:
        margin = tf.random.normal(mean=mean_margin, stddev=.045, shape=())
        margin = tf.where(margin<0, min_margin, margin)
    else:
        margin = mean_margin
    
    # filter out low confidence predictions 
    above_thresh = probas>confidence_thres
    x_coords = tf.boolean_mask(c_kpts[:, 0], above_thresh)
    y_coords = tf.boolean_mask(c_kpts[:, 1], above_thresh)
    
    # get keypoints extrema
    Xmin, Xmax = tf.math.reduce_min(x_coords), tf.math.reduce_max(x_coords)
    Ymin, Ymax = tf.math.reduce_min(y_coords), tf.math.reduce_max(y_coords)

    if Xmax <= Xmin or Ymax <= Ymin:
        Xmax, Xmin = 1., 0.
        Ymax, Ymin = 1., 0.

    c_kpts = tf.convert_to_tensor((Xmin, Ymin, Xmax, Ymax))
    c_kpts += (-margin, -margin, margin, margin)

    # pad coords
    c_kpts = tf.where(c_kpts<0, 0., c_kpts)
    c_kpts = tf.where(c_kpts>1, 1., c_kpts)
    
    # get new dimension to compute the rescale factor
    new_h, new_w = c_kpts[-1]-c_kpts[1], c_kpts[2]-c_kpts[0]

    if new_h < .1 or new_w < .1:
        c_kpts = tf.convert_to_tensor((0., 0., 1., 1.))

    # transform labels
    x_labels = labels[:, 0]
    y_labels = labels[:, 1]
    centre_x = centres[:, 0]
    centre_y = centres[:, 1]

    # shift origin
    x_labels -= c_kpts[0] 
    y_labels -= c_kpts[1] 
    centre_x -= c_kpts[0]
    centre_y -= c_kpts[1]

    # rescale
    x_labels /= new_w
    y_labels /= new_h
    centre_x /= new_w
    centre_y /= new_h

    x_labels = tf.expand_dims(x_labels, axis=-1)
    y_labels = tf.expand_dims(y_labels, axis=-1)
    centre_x = tf.expand_dims(centre_x, axis=-1)
    centre_y = tf.expand_dims(centre_y, axis=-1)    

    centres_mask = tf.math.logical_or(
        tf.math.logical_and(centre_x > 0, centre_x < new_w),
        tf.math.logical_and(centre_y > 0, centre_y < new_h)
        )[:, 0]

    new_labels = tf.concat([x_labels, y_labels], axis=-1)
    new_labels *= tf.expand_dims(probas, axis=1)

    new_centres = tf.concat([centre_x, centre_y], axis=-1)
    new_centres = tf.boolean_mask(new_centres,  centres_mask)
    
    # convert to pixels
    H, W, _ = _ImageDimensions(img, 3)
    c_kpts *= (W, H, W, H)    
    c_kpts = tf.cast(c_kpts, tf.int32)

    img_crop = img[c_kpts[1]:c_kpts[3], c_kpts[0]:c_kpts[2]]

    return img_crop, new_labels, new_centres


