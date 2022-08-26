
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union


from Common.transforms.mask_process import extract_bbox


class BasePatchData(ABC):
    """
    Base class for process patch data pipeline.
    """
    def __init__(self):
        pass

    @abstractmethod
    def split(self, *args: Any, **kwargs: Any):
        """Split image into patches.
        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    def combine(self, *args: Any, **kwargs: Any):
        """Merge result from patches.
        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")


class RandomPatchData(BasePatchData):
    """Generate patch image by random sample mode.
    """
    def __init__(self, patch_radius, fg_ratio: bool = 0.5):
        super(RandomPatchData, self).__init__()
        if not isinstance(patch_radius, list):
            patch_radius = [patch_radius, patch_radius, patch_radius]
        self.patch_radius = patch_radius
        self.fg_ratio = fg_ratio

    def split(self, image, num_patch, fg_mask: Optional[np.ndarray] = None):
        image_depth, image_height, image_width = image.shape
        if fg_mask is not None:
            bbox = extract_bbox(fg_mask)
            min_d, max_d = max(bbox[0, 0], self.patch_radius[0]), \
                           min(bbox[0, 1], image_depth - self.patch_radius[0])
            min_h, max_h = max(bbox[1, 0], self.patch_radius[1]),\
                           min(bbox[1, 1], image_height - self.patch_radius[1])
            min_w, max_w = max(bbox[2, 0], self.patch_radius[2]), \
                           min(bbox[2, 1], image_width - self.patch_radius[2])
        else:
            min_d, max_d = self.patch_radius[0], image_depth-self.patch_radius[0]
            min_h, max_h = self.patch_radius[1], image_height-self.patch_radius[1]
            min_w, max_w = self.patch_radius[2], image_width-self.patch_radius[2]

        all_patch = []
        for i in range(num_patch):
            z_centroid = int(np.random.randint(min_d, max_d, 1)[0])
            y_centroid = int(np.random.randint(min_h, max_h, 1)[0])
            x_centroid = int(np.random.randint(min_w, max_w, 1)[0])

            z_min = int(np.max([0, z_centroid-self.patch_radius[0]]))
            y_min = int(np.max([0, y_centroid-self.patch_radius[1]]))
            x_min = int(np.max([0, x_centroid-self.patch_radius[2]]))

            z_max = int(np.min([image_depth, z_centroid+self.patch_radius[0]]))
            y_max = int(np.min([image_height, y_centroid+self.patch_radius[1]]))
            x_max = int(np.min([image_width, x_centroid+self.patch_radius[2]]))
            bbox = np.array([[z_min, z_max], [y_min, y_max], [x_min, x_max]])
            all_patch.append(bbox)

        return all_patch

    def combine(self):
        pass


class RoiPatchData(BasePatchData):
    """Generate patch image according target rois.
    """
    def __init__(self, patch_radius: Union[List, int], shrink_size: Union[List, int], roi_type: bool = 'centroid'):
        """
        Generate patch image according to target roi.
        :param patch_radius: patch radius.
        :param shrink_size: shrink size for combine mask.
        :param roi_type: ['centroid', 'bbox']
        """
        super(RoiPatchData, self).__init__()
        if not isinstance(patch_radius, list):
            patch_radius = [patch_radius, patch_radius, patch_radius]
        if not isinstance(shrink_size, list):
            shrink_size = [shrink_size, shrink_size, shrink_size]
        for i in range(len(patch_radius)):
            if patch_radius[i] < shrink_size[i]:
                raise ValueError(f'shrink_size: {shrink_size[i]} is exceed than patch_radius: {patch_radius[i]}')
        if roi_type not in ['centroid', 'bbox']:
            raise ValueError(f'{roi_type} is invalid!')
        self.patch_radius = patch_radius
        self.shrink_size = shrink_size
        self.roi_type = roi_type

    def split(self, image: np.ndarray, roi_list: List[np.ndarray]):
        """
        :param image: image array.
        :param roi_list: List of roi, [centroid_x, centroid_y, centroid_z] or
                        [[start_x, end_x], [start_y, end_y], [start_z, end_z]]
        :return:
        """
        image_shape = image.shape
        height, width, depth = image_shape
        assert len(image_shape) == 3
        self.image_shape = image_shape

        all_splits = {}
        idx = 0
        for roi in roi_list:
            if self.roi_type == 'centroid':
                bbox = [[max(0, int(roi[0]-self.patch_radius[0])),
                        min(height, int(roi[0]+self.patch_radius[0]))],
                        [max(0, int(roi[1]-self.patch_radius[1])),
                        min(width, int(roi[1]+self.patch_radius[1]))],
                        [max(0, int(roi[2]-self.patch_radius[2])),
                        min(depth, int(roi[2]+self.patch_radius[2]))]]
            else:
                bbox = [[max(0, roi[0, 0]), min(height, roi[0, 1])],
                        [max(0, roi[1, 0]), min(width, roi[1, 1])],
                        [max(0, roi[2, 0]), min(depth, roi[2, 1])]]
            all_splits[idx] = np.array(bbox)
            idx += 1
        self.all_splits = all_splits

        return self.all_splits

    def combine(self, patches: dict, mode: str = 'mean'):
        """
        :param patches: dict of patch, as {'idx': patch}.
        :param mode: merge mode.
        :return:
        """
        if mode not in ['mean']:
            raise ValueError(f'{mode} mode is not invalid!')
        cur_shape = self.image_shape
        output = np.zeros(cur_shape, np.float32)
        count_matrix = np.zeros(cur_shape, np.float32)
        for idx, patch in patches.items():
            split = self.all_splits[idx]
            h_s, h_e = split[0]
            w_s, w_e = split[1]
            d_s, d_e = split[2]
            bbox_size = [h_e-h_s, w_e-w_s, d_e-d_s]
            patch_size = patch.shape
            if patch_size[0] > bbox_size[0] or patch_size[1] > bbox_size[1] or patch_size[2] > bbox_size[2]:
                patch = patch[:bbox_size[0], :bbox_size[1], :bbox_size[2]]

            if max(self.shrink_size) != 0:
                h_s = split[0, 0] + self.shrink_size[0]
                h_e = split[0, 1] - self.shrink_size[0]
                w_s = split[1, 0] + self.shrink_size[1]
                w_e = split[1, 1] - self.shrink_size[1]
                d_s = split[2, 0] + self.shrink_size[2]
                d_e = split[2, 1] - self.shrink_size[2]
                patch = patch[self.shrink_size[0]-1:-self.shrink_size[0],
                              self.shrink_size[1]-1:-self.shrink_size[1],
                              self.shrink_size[2]-1:-self.shrink_size[2]]

            output[h_s:h_e, w_s:w_e, d_s:d_e] += patch
            count_matrix[h_s:h_e, w_s:w_e, d_s:d_e] += 1

        output = output / count_matrix

        return output


class GridPatchData(BasePatchData):
    """Generate patch image by grid sample mode.
    """
    def __init__(self, patch_size: Union[List, int], stride: Union[List, int]):
        """
        :param patch_size: patch size.
        :param stride: sample stride.
        """
        super(GridPatchData, self).__init__()
        if not isinstance(patch_size, list):
            patch_size = [patch_size, patch_size, patch_size]
        if not isinstance(stride, list):
            stride = [stride, stride, stride]
        self.patch_size = patch_size
        self.stride = stride

    def split(self, image: np.ndarray):
        """
        Split image into multi patches by grid sample mode.
        :param image:
        :return:
        """
        image_shape = image.shape
        height, width, depth = image_shape
        assert len(image_shape) == 3
        num_patch = [int(np.ceil(float(image_shape[i] - self.patch_size[i]) / self.stride[i])) for i in range(3)]
        self.image_shape = image_shape
        self.num_patch = num_patch

        idx = 0
        all_splits = {}
        for h in range(num_patch[0] + 1):
            for w in range(num_patch[1] + 1):
                for d in range(num_patch[2] + 1):
                    h_s = h * self.stride[0]  # start
                    h_e = h * self.stride[0] + self.patch_size[0]  # end
                    w_s = w * self.stride[1]
                    w_e = w * self.stride[1] + self.patch_size[1]
                    d_s = d * self.stride[2]
                    d_e = d * self.stride[2] + self.patch_size[2]
                    if h_e > height:
                        h_s = max(0, height - self.patch_size[0])
                        h_e = height
                    if w_e > width:
                        w_s = max(0, width - self.patch_size[1])
                        w_e = width
                    if d_e > depth:
                        d_s = max(0, depth - self.patch_size[2])
                        d_e = depth
                    crop_bbox = np.array([[h_s, h_e], [w_s, w_e], [d_s, d_e]])
                    all_splits[idx] = crop_bbox
                    idx += 1
        self.all_splits = all_splits

        return all_splits, num_patch

    def combine(self, patches: dict, mode: str = 'mean'):
        """
        Combine patches into raw image coords.
        :param patches: Dict of patch, {'idx': patch_mask }
        :param mode: ['mean', 'max']
        :return:
        """
        if mode not in ['mean', 'max', 'vote']:
            raise ValueError(f'{mode} mode is not invalid!')
        cur_shape = self.image_shape

        if mode == 'mean':
            output = np.zeros(cur_shape, np.float32)
            count_matrix = np.zeros(cur_shape, np.float32)
        else:
            output = -1000000 * np.ones(cur_shape, np.float32)

        for idx, crop_bbox in self.all_splits.items():
            h_s, h_e = crop_bbox[0]
            w_s, w_e = crop_bbox[1]
            d_s, d_e = crop_bbox[2]
            bbox_size = [h_e-h_s, w_e-w_s, d_e-d_s]
            patch = patches[idx]
            patch_size = patch.shape
            if patch_size[0] > bbox_size[0] or patch_size[1] > bbox_size[1] or patch_size[2] > bbox_size[2]:
                patch = patch[:bbox_size[0], :bbox_size[1], :bbox_size[2]]
            if mode == 'mean':
                output[h_s:h_e, w_s:w_e, d_s:d_e] += patch
                count_matrix[h_s:h_e, w_s:w_e, d_s:d_e] += 1
            else:
                output[h_s:h_e, w_s:w_e, d_s:d_e] = patch

        if mode == 'mean':
            output = output / count_matrix

        return output


def split(data, patch_size, stride):
    """
    :param data: target data to be splitted into sub-volumes, shape = (D, H, W) \
    :param patch_size: list of input shape, default=[128, 128, 128] \
    :param stride: sliding stride, default=[64, 64, 64] \
    :return: output list of coordinates for the cropped sub-volumes, start-to-end
    """

    if type(stride) is not list:
        stride = [stride, stride, stride]

    if type(patch_size) is not list:
        patch_size = [patch_size, patch_size, patch_size]

    splits = []
    z, h, w = data.shape

    nz = int(np.ceil(float(z - patch_size[0]) / stride[0]))
    nh = int(np.ceil(float(h - patch_size[1]) / stride[1]))
    nw = int(np.ceil(float(w - patch_size[2]) / stride[2]))

    assert (nz * stride[0] + patch_size[0] - z >= 0)
    assert (nh * stride[1] + patch_size[1] - h >= 0)
    assert (nw * stride[2] + patch_size[2] - w >= 0)

    num_zhw = [nz, nh, nw]
    ori_shape = [z, h, w]

    idx = 0
    for iz in range(nz + 1):
        for ih in range(nh + 1):
            for iw in range(nw + 1):
                sz = iz * stride[0]  # start
                ez = iz * stride[0] + patch_size[0]  # end
                sh = ih * stride[1]
                eh = ih * stride[1] + patch_size[1]
                sw = iw * stride[2]
                ew = iw * stride[2] + patch_size[2]
                if ez > z:
                    sz = max(0, z - patch_size[0])
                    ez = z
                if eh > h:
                    sh = max(0, h - patch_size[1])
                    eh = h
                if ew > w:
                    sw = max(0, w - patch_size[2])
                    ew = w
                idcs = [[sz, ez], [sh, eh], [sw, ew], idx]
                splits.append(idcs)
                idx += 1
    splits = np.array(splits)

    return splits, num_zhw, ori_shape


def combine(output, patch_size, stride, mode='average'):
    """
    combine all things together and average overlapping areas of prediction
    cur_info = [[cur_data, cur_splitID, cur_nzhw, cur_shape, cur_origin, cur_spacing]...]
    : param output: list of all coordinates and voxels of sub-volumes
    : param patch_size: shape of the target volume
    : param stride: stride length of sliding window
    return: output_org, combined volume, original size
    return: curorigin, origin of CT
    return: curspacing, spacing of CT
    """
    assert(mode == 'average' or mode == 'base')
    cur_temp = output[0]
    cur_shape = cur_temp[3]
    cur_origin = cur_temp[4]
    cur_spacing = cur_temp[5]

    nz, nh, nw = cur_temp[2][0], cur_temp[2][1], cur_temp[2][2]
    [z, h, w] = cur_shape
    if type(stride) is not list:
        stride = [stride, stride, stride]
    if type(patch_size) is not list:
        patch_size = [patch_size, patch_size, patch_size]

    splits = {}
    for i in range(len(output)):
        cur_info = output[i]
        cur_data = cur_info[0]
        cur_splitID = int(cur_info[1])
        if not (cur_splitID in splits.keys()):
            splits[cur_splitID] = cur_data
        else:
            continue  # only choose one splits

    if mode == 'average':
        output = np.zeros((z, h, w), np.float32)
        count_matrix = np.zeros((z, h, w), np.float32)
    else:
        output = -1000000 * np.ones((z, h, w), np.float32)

    idx = 0
    for iz in range(nz + 1):
        for ih in range(nh + 1):
            for iw in range(nw + 1):
                sz = iz * stride[0]
                ez = iz * stride[0] + patch_size[0]
                sh = ih * stride[1]
                eh = ih * stride[1] + patch_size[1]
                sw = iw * stride[2]
                ew = iw * stride[2] + patch_size[2]
                if ez > z:
                    sz = z - patch_size[0]
                    ez = z
                if eh > h:
                    sh = h - patch_size[1]
                    eh = h
                if ew > w:
                    sw = w - patch_size[2]
                    ew = w

                split = splits[idx]
                if mode == 'average':
                    output[sz:ez, sh:eh, sw:ew] += split
                    count_matrix[sz:ez, sh:eh, sw:ew] += 1
                else:
                    output[sz:ez, sh:eh, sw:ew] = split
                idx += 1

    if mode == 'average':
        output = output / count_matrix
    output_org = output

    return output_org, cur_origin, cur_spacing
