import os


def get_img_path(dataset_path, scan_folder, view_id, lighting=None, mode=''):
    if mode == "train":
        img_path = os.path.join(dataset_path, "Rectified", scan_folder,"rect_{:0>3}_{}_r5000.png".format(view_id + 1, lighting))
    elif mode == "eval":
        img_path = os.path.join(dataset_path, scan_folder, "images", '{:0>8}.jpg'.format(view_id))
    elif mode == "tanks":
        img_path = os.path.join(dataset_path, scan_folder, "images", '{:0>8}.jpg'.format(view_id)) # f'images/{view_id:08d}.jpg'
    else:
        pass    #custom

    return img_path

def get_cam_path(dataset_path, scan_folder, view_id, mode):
    if mode == "train":
        cam_path = os.path.join(dataset_path, 'Cameras', '{:0>8}_cam.txt'.format(view_id))
    elif mode == "eval":
        cam_path = os.path.join(dataset_path, scan_folder, "cams", '{:0>8}_cam.txt'.format(view_id))
    elif mode == "tanks":
        cam_path = os.path.join(dataset_path, scan_folder, "cams_1", '{:0>8}_cam.txt'.format(view_id))
    else:
        pass  # custom

    return cam_path

def get_depth_path(dataset_path, scan_folder, view_id, mode):
    if mode == "train":
        depth_path = os.path.join(dataset_path, "Depths", scan_folder,'depth_map_{:0>4}.pfm'.format(view_id))
    elif mode == "eval":
        pass
    else:
        pass  # custom

    return depth_path

def get_confidenct_path(mode):
    if mode == "eval":
        pass
    else:
        pass  # custom
