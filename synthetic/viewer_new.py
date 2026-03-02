#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Nerfstudio-style orbit viewer for NeuralPrimitiveScene
# - Drag mouse on the render canvas to orbit (yaw/pitch)
# - Scroll wheel on the render canvas to zoom (radius)
# - Right panel: FOV/Resolution/Batch size, checkpoint loading, primitive selection
# - Render on change or continuously when Live is enabled

import os
import math
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import torch
import dearpygui.dearpygui as dpg

from model import NeuralPrimitiveScene  # your project model
from mip360_loader import mip360Dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
scene: Optional[NeuralPrimitiveScene] = None

# Camera state (spherical around target)
target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
radius = 1.0
yaw_deg = 0.0    # around +Y axis
pitch_deg = 40.0  # around +X axis, limited to [-89, 89]
fov_deg = 60.0   # horizontal FOV in degrees
img_w, img_h = 390, 260
batch_size = 131072
num_primitives = 1024

# dataset
data_dir = "/home/xin/data/mipnerf360/bonsai"
dataset = mip360Dataset(data_dir, split='train')

# UI state
live_render = True

# Orbit control state
orbit_active = False  # begin drag on canvas -> True, mouse release -> False

def set_scene(num_primitives: int, ckpt_path: str = ''):
    global scene
    scene = NeuralPrimitiveScene(num_primitives=num_primitives).to(device)
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Loading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location='cpu')
        scene.load_state_dict(ckpt, strict=False)
    else:
        print("No checkpoint provided or found, initializing scene with random weights.")
    scene.eval()

def get_camera_pose() -> Tuple[np.ndarray, np.ndarray]:
    # Returns camera position (3,) and rotation matrix R_c2w (3,3)
    global yaw_deg, pitch_deg, radius, target
    yaw = math.radians(yaw_deg)
    pitch = math.radians(max(-89.0, min(89.0, pitch_deg)))
    # Spherical to Cartesian (y up)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    cam_dir = np.array([cp * sy, sp, cp * cy], dtype=np.float32)  # forward (z forward in world)
    cam_pos = target + (-cam_dir) * radius  # look at target
    # Build rotation: from camera coordinates to world (camera looks +Z in its local)
    forward = (target - cam_pos)
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(up, forward)
    right = right / (np.linalg.norm(right) + 1e-8)
    cam_up = np.cross(forward, right)
    R = np.stack([right, cam_up, forward], axis=1).astype(np.float32)  # 3x3
    return cam_pos.astype(np.float32), R

def generate_rays(w: int, h: int, fov_x_deg: float, cam_pos: np.ndarray, R_c2w: np.ndarray):
    # pinhole rays for given resolution and horizontal fov
    fov = math.radians(fov_x_deg)
    fx = w / (2.0 * math.tan(fov / 2.0))
    fy = fx
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    js = np.arange(h, dtype=np.float32)
    is_ = np.arange(w, dtype=np.float32)
    xv, yv = np.meshgrid(is_, js)
    x = (xv - cx) / fx
    y = -(yv - cy) / fy  # flip to image coords
    z = np.ones_like(x)
    dirs_cam = np.stack([x, y, z], axis=-1)  # H,W,3
    # normalize
    norm = np.linalg.norm(dirs_cam, axis=-1, keepdims=True) + 1e-8
    dirs_cam = dirs_cam / norm
    # rotate to world
    dirs_world = dirs_cam @ R_c2w.T
    o = np.broadcast_to(cam_pos[None, None, :], dirs_world.shape)
    return o.reshape(-1, 3).astype(np.float32), dirs_world.reshape(-1, 3).astype(np.float32)

@torch.no_grad()
def render_current_view() -> np.ndarray:
    global scene, img_w, img_h, fov_deg, batch_size, dataset, num_primitives
    if scene is None:
        return np.zeros((img_h, img_w, 3), dtype=np.uint8)
    cam_pos, R = get_camera_pose()
    rays_o_np, rays_d_np = generate_rays(img_w, img_h, fov_deg, cam_pos, R)
    rays_o = torch.from_numpy(rays_o_np).to(device)
    rays_d = torch.from_numpy(rays_d_np).to(device)

    # # TODO debug load from dataset view
    # img_idx = 0
    # rays_o, rays_d, rgbs = dataset[img_idx]
    # rays_o = rays_o.to(device).view(-1, 3)  # [H*W, 3]
    # rays_d = rays_d.to(device).view(-1, 3)
    # # TODO debug load from dataset view

    mlp_id = np.arange(num_primitives)  # TODO make selectable
    print("num primitives:", len(mlp_id))

    preds = []
    for i in range(0, rays_o.shape[0], int(batch_size)):
        preds.append(scene(rays_o[i:i+int(batch_size)], rays_d[i:i+int(batch_size)], mlp_id=mlp_id))
    pred_rgb = torch.cat(preds, dim=0).clamp(0, 1).view(img_h, img_w, 3).detach().cpu().numpy()
    img = (pred_rgb * 255.0).astype(np.uint8)
    return img

# DearPyGui setup
def setup_gui():
    dpg.create_context()
    with dpg.texture_registry(show=False):
        # placeholder texture; update per render
        data = np.zeros((img_h, img_w, 4), dtype=np.float32)
        dpg.add_dynamic_texture(width=img_w, height=img_h, default_value=data.flatten().tolist(), tag='render_tex')

    with dpg.window(label='NS-like Viewer', width=img_w+360, height=img_h+80, tag='main_win'):
        with dpg.group(horizontal=True):
            with dpg.child_window(width=img_w+8, height=img_h+8):
                dpg.add_image('render_tex', tag='render_img')
            with dpg.child_window(width=320, height=img_h+8):
                dpg.add_text('Model / Render')
                dpg.add_input_text(label='Checkpoint path', default_value='', tag='ckpt_path')
                dpg.add_input_int(label='num_primitives', default_value=num_primitives, tag='num_primitives', callback=lambda s, a, u: set_primitives(a))
                dpg.add_button(label='Load Model', callback=lambda: on_load_model())
                dpg.add_separator()
                dpg.add_checkbox(label='Live', default_value=True, tag='live_toggle', callback=lambda s, a, u: set_live(a))
                dpg.add_slider_int(label='Width', default_value=img_w, min_value=160, max_value=1920, tag='w_slider', callback=on_res_changed)
                dpg.add_slider_int(label='Height', default_value=img_h, min_value=120, max_value=1080, tag='h_slider', callback=on_res_changed)
                dpg.add_slider_float(label='FOV (deg)', default_value=fov_deg, min_value=10.0, max_value=120.0, tag='fov_slider', callback=lambda s, a, u: set_fov(a))
                dpg.add_slider_float(label='Radius', default_value=radius, min_value=0.2, max_value=10.0, tag='radius_slider', callback=lambda s, a, u: set_radius(a))
                dpg.add_slider_float(label='Yaw (deg)', default_value=yaw_deg, min_value=-180.0, max_value=180.0, tag='yaw_slider', callback=lambda s, a, u: set_yaw(a))
                dpg.add_slider_float(label='Pitch (deg)', default_value=pitch_deg, min_value=-89.0, max_value=89.0, tag='pitch_slider', callback=lambda s, a, u: set_pitch(a))
                dpg.add_slider_int(label='Batch', default_value=batch_size, min_value=32768, max_value=1048576, tag='batch_slider', callback=lambda s, a, u: set_batch(a))
                dpg.add_button(label='Render Once', callback=lambda: render_and_update())
                dpg.add_button(label='Save PNG', callback=lambda: save_png())
                dpg.add_text('', tag='info_text')

    # mouse handlers for orbit control
    with dpg.handler_registry():
        dpg.add_mouse_down_handler(callback=on_mouse_down)
        dpg.add_mouse_release_handler(callback=on_mouse_up)
        dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, callback=on_drag)
        dpg.add_mouse_wheel_handler(callback=on_scroll)

    dpg.create_viewport(title='NS-like Orbit Viewer', width=img_w+360, height=img_h+80)
    dpg.setup_dearpygui()
    dpg.show_viewport()

def set_live(val: bool):
    global live_render
    live_render = bool(val)

def set_primitives(val: int):
    global num_primitives
    num_primitives = int(val)
    if live_render:
        render_and_update()

def on_res_changed(sender, app_data, user_data):
    global img_w, img_h
    img_w = int(dpg.get_value('w_slider'))
    img_h = int(dpg.get_value('h_slider'))
    # recreate texture with new size
    with dpg.texture_registry(show=False):
        data = np.zeros((img_h, img_w, 4), dtype=np.float32)
        if dpg.does_item_exist('render_tex'):
            dpg.delete_item('render_tex')
        dpg.add_dynamic_texture(width=img_w, height=img_h, default_value=data.flatten().tolist(), tag='render_tex')
    if dpg.does_item_exist('render_img'):
        dpg.configure_item('render_img', texture_id='render_tex')
    if live_render:
        render_and_update()

def set_fov(val):
    global fov_deg
    fov_deg = float(val)
    if live_render:
        render_and_update()

def set_radius(val):
    global radius
    radius = float(val)
    if live_render:
        render_and_update()

def set_yaw(val):
    global yaw_deg
    yaw_deg = float(val)
    if live_render:
        render_and_update()

def set_pitch(val):
    global pitch_deg
    pitch_deg = float(val)
    if live_render:
        render_and_update()

def set_batch(val):
    global batch_size
    batch_size = int(val)

# ---------- mouse handlers with canvas scoping ----------

def on_mouse_down(sender, app_data):
    # app_data: [button, x, y]
    global orbit_active
    if app_data[0] == dpg.mvMouseButton_Left and dpg.is_item_hovered('render_img'):
        orbit_active = True

def on_mouse_up(sender, app_data):
    # app_data: [button, x, y]
    global orbit_active
    if app_data[0] == dpg.mvMouseButton_Left:
        orbit_active = False

def on_drag(sender, app_data):
    # app_data: [is_dragging, dx, dy]
    if not orbit_active:
        return
    dx, dy = app_data[1], app_data[2]
    global yaw_deg, pitch_deg
    yaw_deg += dx * 0.2
    pitch_deg += -dy * 0.2
    pitch_deg = max(-89.0, min(89.0, pitch_deg))
    dpg.set_value('yaw_slider', yaw_deg)
    dpg.set_value('pitch_slider', pitch_deg)
    if live_render:
        render_and_update()

def on_scroll(sender, app_data):
    # Only zoom if the mouse is over the render canvas
    if not dpg.is_item_hovered('render_img'):
        return
    delta = float(app_data)
    global radius
    radius *= math.pow(0.9, delta)
    radius = max(0.2, min(20.0, radius))
    dpg.set_value('radius_slider', radius)
    if live_render:
        render_and_update()

# ---------- actions ----------

def on_load_model():
    ckpt = dpg.get_value('ckpt_path') or ''
    num_prims = int(dpg.get_value('num_primitives') or num_primitives)
    set_scene(num_primitives=num_prims, ckpt_path=ckpt)
    dpg.set_value('info_text', f'Model ready on {device}. num_primitives={num_prims}')
    if live_render:
        render_and_update()

def render_and_update():
    img = render_current_view()  # HxWx3 uint8
    # dearpygui dynamic texture expects RGBA float list 0..1
    if img.ndim != 3 or img.shape[2] != 3:
        return
    h, w, _ = img.shape
    rgba = np.concatenate([img, 255*np.ones((h, w, 1), dtype=np.uint8)], axis=2).astype(np.float32) / 255.0
    dpg.set_value('render_tex', rgba.flatten().tolist())

def save_png(path: Optional[str] = None):
    if path is None:
        path = 'ns_view.png'
    img = render_current_view()
    Image.fromarray(img).save(path)
    dpg.set_value('info_text', f'Saved {path}')

def main():
    set_scene(num_primitives=1024, ckpt_path='scene_epoch_62000.pt')  # empty by default; load via UI
    setup_gui()
    # initial render
    render_and_update()
    # run event loop
    dpg.start_dearpygui()
    dpg.destroy_context()

if __name__ == '__main__':
    main()
