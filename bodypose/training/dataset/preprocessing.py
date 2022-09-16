from tensorflow.python.ops.image_ops import _ImageDimensions
import tensorflow as tf

MAX_DEPTH = 3000.0

def decode_img(img, img_size):
    """ Returns the padded image and the original aspect ratio (W/H) from bytes """
    img = tf.io.decode_bmp(img, channels=3)
    height, width, _ = _ImageDimensions(img, 3)
    H, W = img_size
    ratio = tf.cast(width/height, dtype=tf.float32)
    return tf.image.resize_with_pad(img, target_height=H, target_width=W), ratio


def load_RGB(imgPath):
    img = tf.io.read_file(imgPath)
    img = tf.io.decode_png(img, channels=3)
    return tf.cast(img, dtype=tf.float32) / 127.5 - 1


def load_depth(depthPath):
    """ Load the depth image. """
    depth = tf.io.read_file(depthPath)
    depth = tf.io.decode_png(depth, dtype=tf.uint16)
    depth = tf.cast(depth, tf.float32)
    depth = tf.where(depth>MAX_DEPTH, 0., depth/MAX_DEPTH)
    return depth


@tf.function
def create_density_map(joints_coords: tf.Tensor,
                       grid_dim: int):
    """ 
        Creates the density map peaked at the joint coords. 
        input: joint_coords: shape(n_joints, 2)
               grid_dim: side dimension of the FPN output
        output: overall PDF + joints PDFs, tf.tensor shape=(n_grid, n_grid, n_joints+1) 
    """
    # create the grid map
    grid = tf.cast(tf.linspace(0, 1, grid_dim+1), tf.float32)
    grid += 1/(grid_dim)/2
    grid = grid[:-1]
    xx, yy = tf.meshgrid(grid, grid)
    
    # create one grid map for each joint
    n_joints = len(joints_coords)
    XX = tf.tile(tf.expand_dims(xx,-1), [1, 1, n_joints])
    YY = tf.tile(tf.expand_dims(yy,-1), [1, 1, n_joints])
    
    # build the joint PDFs
    X, Y = joints_coords[:,0], joints_coords[:,1]
    sigma = 1.0 / grid_dim
    pdfs = tf.math.exp(-(tf.math.pow(XX-X, 2) + tf.math.pow(YY-Y, 2))/(2*tf.math.pow(sigma, 2)))
    pdfs = pdfs / tf.math.reduce_max(pdfs, axis=[0,1])
    pdfs = tf.where(tf.math.is_nan(pdfs), 0.0, pdfs)
    
    # deal with not visible joints
    # if any of the coords is zero then set the joint to not visible
    mask = tf.where(tf.math.reduce_any(joints_coords==0., axis=-1), 0., 1.)
    mask = tf.reshape(mask, (1, 1, n_joints))

    pdfs *= mask
    pdfs = tf.reshape(pdfs, (grid_dim*grid_dim, -1))
    
    return pdfs


@tf.function
def create_depth_map(joints_coords, grid_dim):
    """ 
        Creates the z_coord map for each joint. 
        INPUT: joint_coords: tf.tensor of shape (n_joints, 2)
               grid_dim: side dimension of the FPN output
        OUTPUT: classification labels, tf.tensor of shape (n_grid, n_grid, n_joints+1)
    """
    n_joints = len(joints_coords)
    z_coords = joints_coords[:, -1]
    y = tf.ones((grid_dim, grid_dim, n_joints), dtype=tf.float32) * z_coords
    y = tf.reshape(y, (grid_dim*grid_dim, n_joints))
    
    return y


def create_joint_mask(joints_name, exclude_joints):
    to_rm_len = len(exclude_joints) 
    mask = tf.stack([joints_name]*to_rm_len, axis=0)
    mask = (mask == tf.reshape(exclude_joints, (to_rm_len,1)))
    mask = tf.math.reduce_sum(tf.cast(mask, tf.int16), axis=0)
    return ~tf.cast(mask, tf.bool)


def mask_img(element):
    """ Applies mask to img (RGB or IR) extracted form the depth. 
        Works also with batches of images.
    """
    img, depth = element
    mask = tf.where(depth==0, 0., 1.)
    return (img * mask, depth)
    

def crop_roi(img, depth, coords, probas, labels, min_margin ,max_margin, thresh=0.05,):
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


