import torch
import numpy as np
import mmcv
import math
import copy

EPS = 1e-3

skeletons = ((0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (5, 7),
             (7, 9), (0, 6), (6, 8), (8, 10), (5, 11), (11, 13),
             (13, 15), (6, 12), (12, 14), (14, 16), (11, 12))


def _get_test_clips(num_frames, clip_len):
    """Uniformly sample indices for testing clips.

    Args:
        num_frames (int): The number of frames.
        clip_len (int): The length of the clip.
    """
    seed = 255
    np.random.seed(seed)

    all_inds = []
    num_clips = 1  # num_clips
    for i in range(num_clips):

        old_num_frames = num_frames
        pi = (1, 1)  # p_interval

        ratio = np.random.rand() * (pi[1] - pi[0]) + pi[0]
        num_frames = int(ratio * num_frames)
        off = np.random.randint(old_num_frames - num_frames + 1)

        if num_frames < clip_len:
            start_ind = i if num_frames < num_clips else i * num_frames // num_clips
            inds = np.arange(start_ind, start_ind + clip_len)
        elif clip_len <= num_frames < clip_len * 2:
            basic = np.arange(clip_len)
            inds = np.random.choice(clip_len + 1, num_frames - clip_len, replace=False)
            offset = np.zeros(clip_len + 1, dtype=np.int64)
            offset[inds] = 1
            offset = np.cumsum(offset)
            inds = basic + offset[:-1]
        else:
            bids = np.array([i * num_frames // clip_len for i in range(clip_len + 1)])
            bsize = np.diff(bids)
            bst = bids[:clip_len]
            offset = np.random.randint(bsize)
            inds = bst + offset

        all_inds.append(inds + off)
        num_frames = old_num_frames

    return np.concatenate(all_inds)


def uniform_sampling(clip_len, keypoints):
    # clip_len = 48
    num_frames = keypoints.shape[1]
    inds = _get_test_clips(num_frames=num_frames, clip_len=clip_len)

    inds = np.mod(inds, num_frames)
    num_person = keypoints.shape[0]
    num_persons = [num_person] * num_frames

    for i in range(num_frames):
        j = num_person - 1
        while j >= 0 and np.all(np.abs(keypoints[j, i]) < 1e-5):
            j -= 1
        num_persons[i] = j + 1
    transitional = [False] * num_frames
    for i in range(1, num_frames - 1):
        if num_persons[i] != num_persons[i - 1]:
            transitional[i] = transitional[i - 1] = True
        if num_persons[i] != num_persons[i + 1]:
            transitional[i] = transitional[i + 1] = True
    inds_int = inds.astype(np.int)
    # print(len(transitional))
    # print(max(inds_int))
    coeff = np.array([transitional[i] for i in inds_int])
    inds = (coeff * inds_int + (1 - coeff) * inds).astype(np.float32)

    return inds.astype(np.int)


def _load_kp(kp, frame_inds):
    return kp[:, frame_inds].astype(np.float32)


def _load_kpscore(kpscore, frame_inds):
    return kpscore[:, frame_inds].astype(np.float32)


def pose_decode(inds, keypoints, keypoints_score):
    # if 'frame_inds' not in results:
    #     results['frame_inds'] = np.arange(results['total_frames'])

    # if results['frame_inds'].ndim != 1:
    #     results['frame_inds'] = np.squeeze(results['frame_inds'])

    # offset = results.get('offset', 0)
    # frame_inds = results['frame_inds'] + offset#-1 #vh_note
    # print(frame_inds)
    # print(results['keypoint_score'])
    # if 'keypoint_score' in results:
    keypoints_score = _load_kpscore(keypoints_score, inds)

    keypoints = _load_kp(keypoints, inds)

    return keypoints, keypoints_score


def _combine_quadruple(a, b):
    return (a[0] + a[2] * b[0], a[1] + a[3] * b[1], a[2] * b[2], a[3] * b[3])


def pose_compact(hw_ratio, allow_imgpad, padding, threshold, img_shape, keypoints):
    # a= keypoints
    # print(keypoints)
    h, w = img_shape
    kp = keypoints

    # Make NaN zero
    kp[np.isnan(kp)] = 0.
    kp_x = kp[..., 0]
    # print(kp_x)
    kp_y = kp[..., 1]
    # print(kp_y)
    min_x = np.min(kp_x[kp_x != 0], initial=np.Inf)
    min_y = np.min(kp_y[kp_y != 0], initial=np.Inf)
    max_x = np.max(kp_x[kp_x != 0], initial=-np.Inf)
    max_y = np.max(kp_y[kp_y != 0], initial=-np.Inf)
    # print(max_x, max_y, min_x, min_y)
    # The compact area is too small
    if max_x - min_x < threshold or max_y - min_y < threshold:
        return None

    center = ((max_x + min_x) / 2, (max_y + min_y) / 2)
    half_width = (max_x - min_x) / 2 * (1 + padding)
    half_height = (max_y - min_y) / 2 * (1 + padding)
    # print(max_x, max_y, min_x, min_y)
    if hw_ratio is not None:
        half_height = max(hw_ratio[0] * half_width, half_height)
        half_width = max(1 / hw_ratio[1] * half_height, half_width)

    min_x, max_x = center[0] - half_width, center[0] + half_width
    min_y, max_y = center[1] - half_height, center[1] + half_height
    # print(max_x, max_y, min_x, min_y)
    # hot update
    if not allow_imgpad:
        min_x, min_y = int(max(0, min_x)), int(max(0, min_y))
        max_x, max_y = int(min(w, max_x)), int(min(h, max_y))
    else:
        min_x, min_y = int(min_x), int(min_y)
        max_x, max_y = int(max_x), int(max_y)
    # print(max_x, max_y, min_x, min_y)
    # print(min_x, min_y)
    kp_x[kp_x != 0] -= min_x
    kp_y[kp_y != 0] -= min_y

    new_shape = (max_y - min_y, max_x - min_x)
    # results['img_shape'] = new_shape

    # the order is x, y, w, h (in [0, 1]), a tuple
    # crop_quadruple = results.get('crop_quadruple', (0., 0., 1., 1.))
    crop_quadruple = (0., 0., 1., 1.)
    new_crop_quadruple = (min_x / w, min_y / h, (max_x - min_x) / w,
                          (max_y - min_y) / h)
    crop_quadruple = _combine_quadruple(crop_quadruple, new_crop_quadruple)
    # print(np.sum(keypoints-a))
    return new_shape, crop_quadruple


def _resize_kps(kps, scale_factor):
    # print(kps)
    # print(scale_factor)
    # print(kps*scale_factor)
    return kps * scale_factor


def resize(scale_factor, keypoints, img_shape, keep_ratio, scale=(64, 64)):
    """Performs the Resize augmentation.

    Args:
        results (dict): The resulting dict to be modified and passed
            to the next transform in pipeline.
    """
    # if 'scale_factor' not in results:
    #     results['scale_factor'] = np.array([1, 1], dtype=np.float32)
    if scale_factor is None:
        scale_factor = np.array([1, 1], dtype=np.float32)
    img_h, img_w = img_shape  # results['img_shape']

    if keep_ratio:
        new_w, new_h = mmcv.rescale_size((img_w, img_h), scale)
    else:
        new_w, new_h = scale

    __scale_factor = np.array([new_w / img_w, new_h / img_h],
                              dtype=np.float32)

    img_shape = (new_h, new_w)
    # results['keep_ratio'] = self.keep_ratio
    sf = scale_factor * __scale_factor

    # if 'imgs' in results:
    #     results['imgs'] = self._resize_imgs(results['imgs'], new_w, new_h)
    if keypoints is not None:
        keypoints = _resize_kps(keypoints, __scale_factor)

    # if 'gt_bboxes' in results:
    #     results['gt_bboxes'] = self._box_resize(results['gt_bboxes'], __scale_factor)
    #     if 'proposals' in results and results['proposals'] is not None:
    #         assert results['proposals'].shape[1] == 4
    #         results['proposals'] = self._box_resize(
    #             results['proposals'], __scale_factor)

    return keep_ratio, sf, keypoints, img_shape


def generate_a_heatmap(arr, centers, max_values, sigma):
    """Generate pseudo heatmap for one keypoint in one frame.

    Args:
        arr (np.ndarray): The array to store the generated heatmaps. Shape: img_h * img_w.
        centers (np.ndarray): The coordinates of corresponding keypoints (of multiple persons). Shape: M * 2.
        max_values (np.ndarray): The max values of each keypoint. Shape: M.

    Returns:
        np.ndarray: The generated pseudo heatmap.
    """

    # sigma = sigma
    img_h, img_w = arr.shape

    for center, max_value in zip(centers, max_values):
        if max_value < EPS:
            continue

        mu_x, mu_y = center[0], center[1]
        # print("muxy", mu_x, mu_y)
        st_x = max(int(mu_x - 3 * sigma), 0)
        ed_x = min(int(mu_x + 3 * sigma) + 1, img_w)
        st_y = max(int(mu_y - 3 * sigma), 0)
        ed_y = min(int(mu_y + 3 * sigma) + 1, img_h)
        x = np.arange(st_x, ed_x, 1, np.float32)

        y = np.arange(st_y, ed_y, 1, np.float32)

        # if the keypoint not in the heatmap coordinate system
        if not (len(x) and len(y)):
            continue
        y = y[:, None]
        # print(y)
        # print(x-mu_x)
        # print(y-mu_y)
        # a=-((x - mu_x)**2 + (y - mu_y)**2) / 2 / sigma**2
        # print(a)
        patch = np.exp(-((x - mu_x) ** 2 + (y - mu_y) ** 2) / 2 / sigma ** 2)
        # print(patch.shape)
        # print(patch)

        patch = patch * max_value
        # print(patch)
        # brr = np.zeros((64,64))
        # for i in range (64):
        #     for j in range (64):
        #         if j >= st_x and j< ed_x and i >= st_y and i < ed_y:
        #             brr[i,j] = np.exp(-((j-mu_x)**2 + (i-mu_y)**2)/2/sigma**2)*max_value
        # print(np.exp(-((j-mu_x)**2 + (i-mu_y)**2)/2/sigma**2)*max_value)
        arr[st_y:ed_y, st_x:ed_x] = np.maximum(arr[st_y:ed_y, st_x:ed_x], patch)

        # for i in range (64):
        #     for j in range (64):
        #         if j >= st_x and j< ed_x and i >= st_y and i < ed_y:
        #             print(brr[i,j],arr[i,j])
        # print(arr.shape)


def generate_a_limb_heatmap(arr, starts, ends, start_values, end_values, sigma):
    """Generate pseudo heatmap for one limb in one frame.

    Args:
        arr (np.ndarray): The array to store the generated heatmaps. Shape: img_h * img_w.
        starts (np.ndarray): The coordinates of one keypoint in the corresponding limbs. Shape: M * 2.
        ends (np.ndarray): The coordinates of the other keypoint in the corresponding limbs. Shape: M * 2.
        start_values (np.ndarray): The max values of one keypoint in the corresponding limbs. Shape: M.
        end_values (np.ndarray): The max values of the other keypoint in the corresponding limbs. Shape: M.

    Returns:
        np.ndarray: The generated pseudo heatmap.
    """

    # sigma = self.sigma
    img_h, img_w = arr.shape

    for start, end, start_value, end_value in zip(starts, ends, start_values, end_values):
        value_coeff = min(start_value, end_value)
        if value_coeff < EPS:
            continue

        min_x, max_x = min(start[0], end[0]), max(start[0], end[0])
        min_y, max_y = min(start[1], end[1]), max(start[1], end[1])

        min_x = max(int(min_x - 3 * sigma), 0)
        max_x = min(int(max_x + 3 * sigma) + 1, img_w)
        min_y = max(int(min_y - 3 * sigma), 0)
        max_y = min(int(max_y + 3 * sigma) + 1, img_h)

        x = np.arange(min_x, max_x, 1, np.float32)
        y = np.arange(min_y, max_y, 1, np.float32)

        if not (len(x) and len(y)):
            continue

        y = y[:, None]
        x_0 = np.zeros_like(x)
        y_0 = np.zeros_like(y)

        # distance to start keypoints
        d2_start = ((x - start[0]) ** 2 + (y - start[1]) ** 2)

        # distance to end keypoints
        d2_end = ((x - end[0]) ** 2 + (y - end[1]) ** 2)

        # the distance between start and end keypoints.
        d2_ab = ((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2)

        if d2_ab < 1:
            generate_a_heatmap(arr, start[None], start_value[None], sigma=sigma)
            continue

        coeff = (d2_start - d2_end + d2_ab) / 2. / d2_ab

        a_dominate = coeff <= 0
        b_dominate = coeff >= 1
        seg_dominate = 1 - a_dominate - b_dominate

        position = np.stack([x + y_0, y + x_0], axis=-1)
        projection = start + np.stack([coeff, coeff], axis=-1) * (end - start)
        d2_line = position - projection
        d2_line = d2_line[:, :, 0] ** 2 + d2_line[:, :, 1] ** 2
        d2_seg = a_dominate * d2_start + b_dominate * d2_end + seg_dominate * d2_line

        patch = np.exp(-d2_seg / 2. / sigma ** 2)
        patch = patch * value_coeff

        arr[min_y:max_y, min_x:max_x] = np.maximum(arr[min_y:max_y, min_x:max_x], patch)


def generate_heatmap(arr, kps, max_values, with_kps, with_limb, skeletons, sigma):
    """Generate pseudo heatmap for all keypoints and limbs in one frame (if
    needed).

    Args:
        arr (np.ndarray): The array to store the generated heatmaps. Shape: V * img_h * img_w.
        kps (np.ndarray): The coordinates of keypoints in this frame. Shape: M * V * 2.
        max_values (np.ndarray): The confidence score of each keypoint. Shape: M * V.

    Returns:
        np.ndarray: The generated pseudo heatmap.
    """

    if with_kps:
        num_kp = kps.shape[1]
        for i in range(num_kp):
            generate_a_heatmap(arr[i], kps[:, i], max_values[:, i], sigma=sigma)

    if with_limb:
        for i, limb in enumerate(skeletons):
            start_idx, end_idx = limb
            starts = kps[:, start_idx]
            ends = kps[:, end_idx]

            start_values = max_values[:, start_idx]
            end_values = max_values[:, end_idx]
            generate_a_limb_heatmap(arr[i], starts, ends, start_values, end_values, sigma=sigma)


def gen_an_aug(keypoints, keypoints_score, img_shape, with_kps, with_limb, skeletons, use_score, sigma):
    """Generate pseudo heatmaps for all frames.

    Args:
        results (dict): The dictionary that contains all info of a sample.

    Returns:
        list[np.ndarray]: The generated pseudo heatmaps.
    """

    all_kps = keypoints
    kp_shape = all_kps.shape

    if keypoints_score is not None:
        all_kpscores = keypoints_score
    else:
        all_kpscores = np.ones(kp_shape[:-1], dtype=np.float32)

    img_h, img_w = img_shape
    num_frame = kp_shape[1]
    num_c = 0
    if with_kps:
        num_c += all_kps.shape[2]
    if with_limb:
        num_c += len(skeletons)
    ret = np.zeros([num_frame, num_c, img_h, img_w], dtype=np.float32)

    for i in range(num_frame):
        # M, V, C
        kps = all_kps[:, i]
        # M, C
        kpscores = all_kpscores[:, i] if use_score else np.ones_like(all_kpscores[:, i])

        generate_heatmap(ret[i], kps, kpscores, with_kps=with_kps, with_limb=with_limb, skeletons=skeletons,
                         sigma=sigma)
    return ret


def generate_target(keypoints, keypoints_score, img_shape, with_kps, with_limb, skeletons, use_score, sigma):
    # heatmap = gen_an_aug(results)
    # hm = gen_an_aug(keypoints, keypoints_score, img_shape, with_kps, with_limb, skeletons, use_score, sigma=sigma)
    heatmap = np.zeros((48, 17, 64, 64))

    # print(keypoints.shape)
    # print(keypoints_score.shape)
    for i in range(48):
        for j in range(17):
            img_h = img_shape[0]
            # print(img_shape)
            img_w = img_shape[1]

            mu_x = keypoints[0][i][j][0]

            mu_y = keypoints[0][i][j][1]
            # print(mu_x,mu_y)
            st_x = max(int(mu_x - 3 * sigma), 0)
            ed_x = min(int(mu_x + 3 * sigma) + 1, img_w)
            st_y = max(int(mu_y - 3 * sigma), 0)
            ed_y = min(int(mu_y + 3 * sigma) + 1, img_h)

            for k in range(img_h):
                for m in range(img_w):
                    if (m >= st_x and m < ed_x and k >= st_y and k < ed_y):
                        heatmap[i][j][k][m] = math.exp(-((m - mu_x) ** 2 + (k - mu_y) ** 2) / 2 / sigma ** 2) * \
                                              keypoints_score[0][i][j]

    return heatmap


def format_shape(heatmap, collapse, input_format, num_clips, clip_len):
    """Performs the FormatShape formatting.

    Args:
        results (dict): The resulting dict to be modified and passed
            to the next transform in pipeline.
    """
    if not isinstance(heatmap, np.ndarray):
        heatmap = np.array(heatmap)
    imgs = heatmap
    # [M x H x W x C]
    # M = 1 * N_crops * N_clips * L
    if collapse:
        assert num_clips == 1
    # my_imgs = copy.deepcopy(heatmap)
    my_imgs = np.zeros((17, 48, 64, 64))

    if input_format == 'NCTHW':
        num_clips = num_clips  # results['num_clips']
        clip_len = clip_len  # results['clip_len']

        imgs = imgs.reshape((-1, num_clips, clip_len) + imgs.shape[1:])
        # N_crops x N_clips x L x H x W x C
        imgs = np.transpose(imgs, (0, 1, 5, 2, 3, 4))
        # N_crops x N_clips x C x L x H x W
        imgs = imgs.reshape((-1,) + imgs.shape[2:])
        # M' x C x L x H x W
        # M' = N_crops x N_clips
    elif input_format == 'NCTHW_Heatmap':
        num_clips = num_clips  # results['num_clips']
        clip_len = clip_len  # results['clip_len']
        # print(imgs.shape)
        imgs = imgs.reshape((-1, num_clips, clip_len) + imgs.shape[1:])
        # N_crops x N_clips x L x C x H x W
        # print(imgs.shape)
        imgs = np.transpose(imgs, (0, 1, 3, 2, 4, 5))
        # print(imgs.shape)
        # N_crops x N_clips x C x L x H x W
        imgs = imgs.reshape((-1,) + imgs.shape[2:])
        # print(imgs.shape)
        # M' x C x L x H x W
        # M' = N_crops x N_clips
        # print(np.expand_dims(my_imgs,0))
        # print(np.sum(np.expand_dims(my_imgs,0)- imgs))
    elif input_format == 'NCHW':
        imgs = np.transpose(imgs, (0, 3, 1, 2))
        # M x C x H x W

    if collapse:
        assert imgs.shape[0] == 1
        imgs = imgs.squeeze(0)

    # results['imgs'] = imgs
    # results['input_shape'] = imgs.shape
    return imgs, imgs.shape


# def collect():

def vh_uniform_sampling(clip_len, keypoints):
    inds = []
    num_frames = keypoints.shape[1]
    if num_frames < clip_len:
        for i in range(clip_len):
            inds.append(i % num_frames)
    else:
        for i in range(clip_len):
            inds.append(round(i * num_frames / clip_len))

    return np.array(inds)


def pipeline(keypoints, keypoints_score, img_shape):
    clip_len = 48

    # inds = uniform_sampling(clip_len=clip_len,keypoints=keypoints)
    inds = vh_uniform_sampling(clip_len=clip_len, keypoints=keypoints)

    kps, kps_scr = pose_decode(inds=inds, keypoints=keypoints, keypoints_score=keypoints_score)

    hw_ratio = (1., 1.)
    allow_imgpad = True
    padding = 0.25
    threshold = 10

    # import copy
    # a= copy.deepcopy(kps)
    new_shape, crop_quadruple = pose_compact(hw_ratio=hw_ratio, allow_imgpad=allow_imgpad, padding=padding,
                                             threshold=threshold, img_shape=img_shape, keypoints=kps)
    # print(new_shape)
    # print("hello")
    # print(kps-a)
    # print(new_shape)

    keep_ratio, sf, rs_kps, new_shape = resize(scale_factor=None, keypoints=kps, img_shape=new_shape, keep_ratio=True,
                                               scale=(64, 64))

    heatmap = generate_target(keypoints=rs_kps, keypoints_score=kps_scr, img_shape=new_shape, with_kps=True,
                              with_limb=False, skeletons=skeletons, use_score=True, sigma=0.6)

    input, input_shape = format_shape(heatmap=heatmap, collapse=False, input_format='NCTHW_Heatmap', num_clips=1,
                                      clip_len=clip_len)

    return input, input_shape