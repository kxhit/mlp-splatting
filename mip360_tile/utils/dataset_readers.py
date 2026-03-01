import os, sys
import numpy as np
from typing import NamedTuple
from colmap_loader import (
    read_extrinsics_text, read_intrinsics_text, qvec2rotmat,
    read_extrinsics_binary, read_intrinsics_binary
)
from graphics_utils import focal2fov

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: float
    FovX: float
    image_path: str
    image_name: str
    width: int
    height: int
    is_test: bool

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, test_cam_names_list):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write(f"\rReading camera {idx+1}/{len(cam_extrinsics)}")
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]

        height, width = intr.height, intr.width
        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal = intr.params[0]
            FovY = focal2fov(focal, height)
            FovX = focal2fov(focal, width)
        elif intr.model == "PINHOLE":
            fx, fy = intr.params[0], intr.params[1]
            FovX, FovY = focal2fov(fx, width), focal2fov(fy, height)
        else:
            raise ValueError(f"Unsupported COLMAP model {intr.model}")

        image_path = os.path.join(images_folder, extr.name)
        cam_info = CameraInfo(
            uid=uid, R=R, T=T, FovY=FovY, FovX=FovX,
            image_path=image_path, image_name=extr.name,
            width=width, height=height,
            is_test=extr.name in test_cam_names_list
        )
        cam_infos.append(cam_info)
    sys.stdout.write("\n")
    return cam_infos

def readColmapSceneInfo(path, images, eval=True, llffhold=8):
    try:
        cam_extrinsics = read_extrinsics_binary(os.path.join(path, "images.bin"))
        cam_intrinsics = read_intrinsics_binary(os.path.join(path, "cameras.bin"))
    except:
        cam_extrinsics = read_extrinsics_text(os.path.join(path, "images.txt"))
        cam_intrinsics = read_intrinsics_text(os.path.join(path, "cameras.txt"))

    # test split
    if eval:
        cam_names = sorted([cam_extrinsics[cam_id].name for cam_id in cam_extrinsics])
        test_cam_names_list = [name for idx, name in enumerate(cam_names) if idx % llffhold == 0]
    else:
        test_cam_names_list = []

    images_folder = os.path.join(path, "../images") if images is None else images
    cam_infos = readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, test_cam_names_list)
    cam_infos = sorted(cam_infos, key=lambda x: x.image_name)

    train_cam_infos = [c for c in cam_infos if not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]

    return {
        "train_cameras": train_cam_infos,
        "test_cameras": test_cam_infos
    }

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo
}
