import re
import numpy as np


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
        header = fp.readline().rstrip()
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        # grab image dimensions
        dim_match = re.match(r'^(\d+)\s(\d+)\s$', fp.readline())
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')

        # grab image scale
        scale = float(fp.readline().rstrip())
        if scale < 0:  # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>'  # big-endian

        # grab image data
        data = np.fromfile(fp, endian + 'f')
        shape = (height, width, 3) if color else (height, width)

        # reshape data to [Height, Width, Channels]
        data = np.reshape(data, shape)
        data = np.flipud(data)

        return data, scale


# load utils
def load_scene_flow_disp(img_path):
    """load scene flow disparity image
    Args:
        img_path:
    Returns:
    """
    assert img_path.endswith('.pfm'), "scene flow disparity image must end with .pfm" \
                                      "but got {}".format(img_path)

    disp_img, __ = load_pfm(img_path)

    return disp_img
