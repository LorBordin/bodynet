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
    sigma = 1.0 / grid_dim
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
            Multi peaked density map of shape (grid_dim ** 2).
    """
    pdf = tf.math.reduce_sum(pdfs, axis=-1)
    pdf = tf.where(pdf>1.0, 1.0, pdf)
    return pdf


def crop_roi(
        img, 
        depth, 
        coords, 
        probas, 
        labels, 
        min_margin,
        max_margin, 
        thresh=0.05,
    ):
    """ Assume coords are in the shape (-1, 3) and img in format [H, W, C]""" 
    margin = tf.random.normal(mean=0.15, stddev=.045, shape=())
    margin = tf.where(margin<0, .01, margin)
    
    above_thresh = probas>thresh
    x_coords = tf.boolean_mask(coords[:, 0], above_thresh)
    y_coords = tf.boolean_mask(coords[:, 1], above_thresh)

    Xmin, Xmax = tf.math.reduce_min(x_coords), tf.math.reduce_max(x_coords)
    Ymin, Ymax = tf.math.reduce_min(y_coords), tf.math.reduce_max(y_coords)

    if Xmax<=Xmin or Ymax<=Ymin:
        Xmax, Xmin = 1., 0.
        Ymax, Ymin = 1., 0.

    coords = tf.convert_to_tensor((Xmin, Ymin, Xmax, Ymax))
    coords += (-margin, -margin, margin, margin)

    # pad coords
    coords = tf.where(coords<0, 0., coords)
    coords = tf.where(coords>1, 1., coords)
    
    # get new dimension to compute the rescale factor
    new_h, new_w = coords[-1]-coords[1], coords[2]-coords[0]

    if new_h < .1 or new_w < .1:
        coords = tf.convert_to_tensor((0., 0., 1., 1.))

    # transform labels
    x_labels = labels[:, 0]
    y_labels = labels[:, 1]
    z_labels = labels[:, 2]

    # shift origin
    x_labels -= coords[0] 
    y_labels -= coords[1] 
    
    # rescale
    x_labels /= new_w
    y_labels /= new_h

    x_labels = tf.expand_dims(x_labels, axis=-1)
    y_labels = tf.expand_dims(y_labels, axis=-1)
    z_labels = tf.expand_dims(z_labels, axis=-1)

    new_labels = tf.concat([x_labels, y_labels, z_labels], axis=-1)
    
    # convert to pixels
    H, W, _ = _ImageDimensions(img, 3)
    coords *= (W, H, W, H)    
    coords = tf.cast(coords, tf.int32)

    img_crop = img[coords[1]:coords[3], coords[0]:coords[2]]
    depth_crop = depth[coords[1]:coords[3], coords[0]:coords[2]]

    return img_crop, depth_crop, new_labels


