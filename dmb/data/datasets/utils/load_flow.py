import re
import numpy as np
import png


def load_pfm(file_path):
    """
    load image in PFM type.
    Args:
        file_path string: file path(absolute)
    Returns:
        data (numpy.array): data of image in (Height, Width[, 3]) layout
        scale (float): scale of image
    """
    with open(file_path, encoding="ISO-8859-1") as fp:

        color = None
        width = None
        height = None
        scale = None
        endian = None

        # load file header and grab channels, if is 'PF' 3 channels else 1 channel(gray scale)
        header = fp.readline().rstrip().decode('utf-8')
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', fp.readline().decode('utf-8'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')

        scale = float(fp.readline().rstrip())
        if scale < 0: # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>' # big-endian

        data = np.fromfile(fp, endian + 'f')
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)

    return data, scale


def load_png(file_path):
    """
    Read from KITTI .png file
    Args:
        file_path string: file path(absolute)
    Returns:
        data (numpy.array): data of image in (Height, Width, 3) layout
    """
    flow_object = png.Reader(filename=file_path)
    flow_direct = flow_object.asDirect()
    flow_data = list(flow_direct[2])
    (w, h) = flow_direct[3]['size']

    flow = np.zeros((h, w, 3), dtype=np.float64)
    for i in range(len(flow_data)):
        flow[i, :, 0] = flow_data[i][0::3]
        flow[i, :, 1] = flow_data[i][1::3]
        flow[i, :, 2] = flow_data[i][2::3]

    invalid_idx = (flow[:, :, 2] == 0)
    flow[:, :, 0:2] = (flow[:, :, 0:2] - 2 ** 15) / 64.0
    flow[invalid_idx, 0] = 0
    flow[invalid_idx, 1] = 0

    return flow.astype(np.float32)


def load_flo(file_path):
    """
    Read .flo file in MiddleBury format
    Code adapted from:
    http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    WARNING: this will work on little-endian architectures (eg Intel x86) only!
    Args:
        file_path string: file path(absolute)
    Returns:
        flow (numpy.array): data of image in (Height, Width, 2) layout
    """

    with open(file_path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        assert(magic == 202021.25)
        w = int(np.fromfile(f, np.int32, count=1))
        h = int(np.fromfile(f, np.int32, count=1))
        # print('Reading %d x %d flo file\n' % (w, h))
        flow = np.fromfile(f, np.float32, count=2 * w * h)
        # Reshape data into 3D array (columns, rows, bands)
        # The reshape here is for visualization, the original code is (w,h,2)
        flow = np.resize(flow, (h, w, 2))

    return flow


def write_flo(file_path, uv, v=None):
    """ Write optical flow to file.

    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2

    if v is None:
        assert (uv.ndim == 3)
        assert (uv.shape[2] == 2)
        u = uv[:, :, 0]
        v = uv[:, :, 1]
    else:
        u = uv

    assert (u.shape == v.shape)
    height, width = u.shape
    f = open(file_path, 'wb')
    # write the header
    np.array([202021.25]).astype(np.float32).tofile(f)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width * nBands))
    tmp[:, np.arange(width) * 2] = u
    tmp[:, np.arange(width) * 2 + 1] = v
    tmp.astype(np.float32).tofile(f)

    f.close()


# load utils
def load_flying_chairs_flow(img_path):
    """load flying chairs flow image
    Args:
        img_path:
    Returns:
    """
    assert img_path.endswith('.flo'), "flying chairs flow image must end with .flo " \
                                      "but got {}".format(img_path)

    flow_img = load_flo(img_path)

    return flow_img


# load utils
def write_flying_chairs_flow(img_path, uv, v=None):
    """write flying chairs flow image
    Args:
        img_path:
    Returns:
    """
    assert img_path.endswith('.flo'), "flying chairs flow image must end with .flo " \
                                      "but got {}".format(img_path)

    write_flo(img_path, uv, v)


# load utils
def load_flying_things_flow(img_path):
    """load flying things flow image
    Args:
        img_path:
    Returns:
    """
    assert img_path.endswith('.pfm'), "flying things flow image must end with .pfm " \
                                      "but got {}".format(img_path)

    flow_img, __ = load_pfm(img_path)

    return flow_img


# load utils
def load_kitti_flow(img_path):
    """load KITTI 2012/2015 flow image
    Args:
        img_path:
    Returns:
    """
    assert img_path.endswith('.png'), "KITTI 2012/2015 flow image must end with .png " \
                                      "but got {}".format(img_path)

    flow_img = load_png(img_path)

    return flow_img
