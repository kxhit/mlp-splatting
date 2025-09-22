import os
import sys
import time
import argparse
import math
import numpy as np
import torch
import torch.nn.functional as F
from dearpygui import dearpygui as dpg

# Support running as a script (python viewer/run_viewer.py) and as a module
try:
	from .camera import OrbitCamera
	from .adapter import build_scene_from_ckpt
	from .renderer import render_scene
except Exception:  # ImportError or runtime packaging issues
	from camera import OrbitCamera
	from adapter import build_scene_from_ckpt
	from renderer import render_scene

# robust import for dataset
try:
	from ..mip360_loader import mip360Dataset  # type: ignore
except Exception:
	try:
		from mip360_loader import mip360Dataset  # type: ignore
	except Exception:
		parent = os.path.dirname(os.path.dirname(__file__))
		if parent not in sys.path:
			sys.path.append(parent)
		from mip360_loader import mip360Dataset  # type: ignore


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--output_dir", type=str, required=True)
	parser.add_argument("--ckpt", type=str, required=True)
	parser.add_argument("--width", type=int, default=800)
	parser.add_argument("--height", type=int, default=600)
	parser.add_argument("--fov", type=float, default=60.0)
	parser.add_argument("--device", type=str, default="cuda")
	parser.add_argument("--batch_size", type=int, default=60000)
	parser.add_argument("--data_dir", type=str, default="")
	parser.add_argument("--factor", type=int, default=8)
	parser.add_argument("--llffhold", type=int, default=8)
	return parser.parse_args()


class ViewerApp:
	def __init__(self, args):
		os.makedirs(args.output_dir, exist_ok=True)
		self.args = args
		self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
		# checkpoint management (file or directory)
		self.ckpt_dir = None
		self.ckpt_items = []  # UI items (filenames)
		self.ckpt_paths = []  # full paths aligned with items
		self.selected_ckpt_path = None
		if os.path.isdir(args.ckpt):
			self.ckpt_dir = args.ckpt
			self._scan_ckpt_dir()
			# choose the most recently modified .pt file if available
			if self.ckpt_paths:
				self.selected_ckpt_path = self.ckpt_paths[0]
			else:
				raise FileNotFoundError(f"No .pt files found in directory: {args.ckpt}")
		else:
			self.selected_ckpt_path = args.ckpt
		self.scene = build_scene_from_ckpt(self.selected_ckpt_path, self.device)

		self.W = args.width
		self.H = args.height
		self.aspect = float(self.W) / float(self.H) if self.H != 0 else 1.0
		# Base render resolution (MLP runs only at this size)
		self.base_W = int(self.W)
		self.base_H = int(self.H)
		self.cam = OrbitCamera(self.base_W, self.base_H, fov_deg=args.fov)
		self.last_time = time.time()
		self.needs_redraw = True
		self.dragging = False
		self.last_mouse = (0, 0)
		self.batch_size = args.batch_size
		self.use_dataset_pose = False
		self.show_point_centers = False
		self.overlay_point_radius_px = 2
		# exclusion selection state
		self.enable_exclude_select = False
		self.excluded_mask = None  # torch.bool on CPU sized to num_primitives
		self._log_scales_backup = None  # backup of original log_scales for restore
		# rectangle selection state (Shift-drag / Ctrl+Shift-drag)
		self._rect_select_active = False
		self._rect_select_start = (0.0, 0.0)
		self._rect_select_end = (0.0, 0.0)
		self._rect_select_include = False  # True: include (ctrl+shift), False: exclude (shift only)
		# track previous cumulative drag deltas per button to compute incremental deltas
		self._left_drag_prev = None  # type: tuple[float, float] | None
		self._right_drag_prev = None  # type: tuple[float, float] | None
		self._did_left_drag = False

		# dataset poses (optional)
		self.dataset = None
		self.dataset_labels = []
		self.dataset_label_to_index = {}
		self.dataset_H = None
		self.dataset_W = None
		self.dataset_focal = None
		self.dataset_cam_idx = None
		if args.data_dir and os.path.isdir(args.data_dir):
			try:
				self.dataset = mip360Dataset(args.data_dir, factor=args.factor, llffhold=args.llffhold, split='train')
				if len(self.dataset.images) > 0:
					img0 = self.dataset.images[0]
					self.dataset_C, self.dataset_H, self.dataset_W = img0.shape
					self.dataset_focal = float(self.dataset.focal)
					# labels: "index (filename)" using dataset-provided paths if available
					try:
						if hasattr(self.dataset, 'image_paths_selected') and isinstance(self.dataset.image_paths_selected, list) and len(self.dataset.image_paths_selected) == len(self.dataset):
							names = [os.path.basename(p) for p in self.dataset.image_paths_selected]
						elif hasattr(self.dataset, 'image_paths') and hasattr(self.dataset, 'select_ids') and len(self.dataset.select_ids) == len(self.dataset):
							names = [os.path.basename(self.dataset.image_paths[i]) for i in self.dataset.select_ids]
						else:
							names = [""] * len(self.dataset)
						self.dataset_labels = [f"{i} ({names[i]})" if names[i] else f"{i}" for i in range(len(self.dataset))]
						self.dataset_label_to_index = {self.dataset_labels[i]: i for i in range(len(self.dataset))}
					except Exception:
						self.dataset_labels = [f"{i}" for i in range(len(self.dataset))]
						self.dataset_label_to_index = {self.dataset_labels[i]: i for i in range(len(self.dataset))}
			except Exception as e:
				print("[viewer] Failed to load dataset poses:", e)

		dpg.create_context()
		dpg.create_viewport(title="MLP Splatting Viewer", width=self.W+320, height=self.H+80)
		# texture registry and initial texture
		# texture registry and initial texture (keep handle to reuse on resize)
		self.tex_reg = dpg.add_texture_registry()
		self.tex_id = dpg.add_dynamic_texture(self.W, self.H, [0.0]*(self.W*self.H*4), parent=self.tex_reg)

		with dpg.window(label="Controls", width=300, height=self.H, pos=(self.W+10, 10)):
			self.yaw_slider = dpg.add_slider_float(label="Yaw", default_value=0.0, min_value=-np.pi, max_value=np.pi, callback=self._on_yaw)
			self.pitch_slider = dpg.add_slider_float(label="Pitch", default_value=0.0, min_value=-1.2, max_value=1.2, callback=self._on_pitch)
			self.radius_slider = dpg.add_slider_float(label="Radius", default_value=self.cam.radius, min_value=0.3, max_value=5.0, callback=self._on_radius)
			self.fov_slider = dpg.add_slider_float(label="FOV", default_value=self.cam.fov_deg, min_value=30.0, max_value=100.0, callback=self._on_fov)
			self.size_slider = dpg.add_slider_int(label="Render Height (px)", default_value=self.H, min_value=200, max_value=1600, callback=self._on_resize_height)
			dpg.add_button(label="Screenshot", callback=self._on_screenshot)
			dpg.add_text(default_value=f"Device: {self.device}")
			# checkpoint dropdown if a directory was provided
			if self.ckpt_dir is not None and self.ckpt_items:
				self.ckpt_combo = dpg.add_combo(label="Checkpoint", items=self.ckpt_items, default_value=os.path.basename(self.selected_ckpt_path), callback=self._on_select_ckpt)
			if self.dataset_labels:
				self.dataset_combo = dpg.add_combo(label="Training View", items=self.dataset_labels, callback=self._on_select_view)
				self.lock_checkbox = dpg.add_checkbox(label="Lock to dataset pose", default_value=False, callback=self._on_toggle_lock_view)
			self.overlay_checkbox = dpg.add_checkbox(label="Show MLP point centers (yellow/red)", default_value=False, callback=self._on_toggle_overlay)
			# Exclusion selection controls
			self.exclude_checkbox = dpg.add_checkbox(label="MLP Exclusion (Ctrl+Click)", default_value=False, callback=self._on_toggle_exclude_mode)
			self.btn_all_set = dpg.add_button(label="all-set", callback=self._on_all_set, enabled=False)
			self.btn_all_reset = dpg.add_button(label="all-reset", callback=self._on_all_reset, enabled=False)
		with dpg.window(label="Render", width=self.W, height=self.H, pos=(10, 10)) as self.render_win:
			with dpg.drawlist(width=self.W, height=self.H) as self.drawlist:
				self.draw_image_id = dpg.draw_image(self.tex_id, (0, 0), (self.W, self.H))

		dpg.set_primary_window(self.render_win, True)
		dpg.setup_dearpygui()
		dpg.show_viewport()

		# input handlers
		with dpg.handler_registry():
			dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, callback=self._on_mouse_drag)
			dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Right, callback=self._on_mouse_drag_right)
			dpg.add_mouse_wheel_handler(callback=self._on_mouse_wheel)
			dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Left, callback=self._on_mouse_click)
			# key handlers for roll: '[' and ']'
			dpg.add_key_press_handler(key=ord('['), callback=self._on_roll_ccw)
			dpg.add_key_press_handler(key=ord(']'), callback=self._on_roll_cw)
			# commit orientation on roll key release (no rebase to global up here to avoid snap-back)
			dpg.add_key_release_handler(key=ord('['), callback=self._on_roll_released)
			dpg.add_key_release_handler(key=ord(']'), callback=self._on_roll_released)
			# A/D dolly forward/backward (uppercase and lowercase)
			dpg.add_key_press_handler(key=ord('W'), callback=self._on_dolly_forward)
			dpg.add_key_press_handler(key=ord('w'), callback=self._on_dolly_forward)
			dpg.add_key_press_handler(key=ord('S'), callback=self._on_dolly_backward)
			dpg.add_key_press_handler(key=ord('s'), callback=self._on_dolly_backward)
			# reset drag accumulators on mouse release
			dpg.add_mouse_release_handler(callback=self._on_mouse_release)

	def _mouse_over_render(self) -> bool:
		try:
			return dpg.is_item_hovered(self.drawlist)
		except Exception:
			return False

	def _on_yaw(self, s, a):
		self.cam.yaw = float(a)
		self.needs_redraw = True

	def _on_pitch(self, s, a):
		self.cam.pitch = float(a)
		self.needs_redraw = True

	def _on_radius(self, s, a):
		self.cam.radius = float(a)
		self.needs_redraw = True

	def _on_fov(self, s, a):
		self.cam.fov_deg = float(a)
		self.needs_redraw = True

	def _on_mouse_release(self, s, a):
		# a may contain button info; regardless, reset drag state
		self._left_drag_prev = None
		self._right_drag_prev = None
		# finalize rectangle selection if active
		if self._rect_select_active:
			try:
				x0, y0 = self._rect_select_start
				x1, y1 = self._rect_select_end
				x_min = min(x0, x1); x_max = max(x0, x1)
				y_min = min(y0, y1); y_max = max(y0, y1)
				# commit selection only if the rectangle has area
				if abs(x_max - x_min) > 1 and abs(y_max - y_min) > 1 and self.show_point_centers:
					self._ensure_exclusion_state()
					if self.excluded_mask is not None and self.excluded_mask.numel() > 0:
						c2w_used = self._current_c2w_for_projection()
						with torch.no_grad():
							points = self.scene.positions
							px, py, valid = self._project_points_to_pixels(points, c2w_used)
							px = px.numpy(); py = py.numpy(); valid = valid.numpy()
							inside = (px >= x_min) & (px <= x_max) & (py >= y_min) & (py <= y_max) & valid
							inside_t = torch.from_numpy(inside)
							if self._rect_select_include:
								# include -> un-exclude
								self.excluded_mask[inside_t] = False
							else:
								# exclude
								self.excluded_mask[inside_t] = True
							# apply or restore depending on mode
							if self.enable_exclude_select:
								self._apply_exclusion_mask_to_scene()
							else:
								self._restore_scene_scales()
			except Exception:
				pass
			# clear selection state
			self._rect_select_active = False
			self.needs_redraw = True
			# do not treat this as an orbit drag
			self._did_left_drag = False
			return
		# if user performed an orbit drag, commit the camera's current up as new yaw axis
		if self._did_left_drag:
			self.cam.commit_up_from_camera()
			self._did_left_drag = False
			try:
				dpg.set_value(self.yaw_slider, self.cam.yaw)
				dpg.set_value(self.pitch_slider, self.cam.pitch)
			except Exception:
				pass
			self.needs_redraw = True

	def _on_mouse_drag(self, s, a):
		if not self._mouse_over_render():
			return
		# If exclusion rectangle selection is active (Shift held), consume the drag for selection
		shift_down = False
		try:
			shift_down = dpg.is_key_down(dpg.mvKey_LShift) or dpg.is_key_down(dpg.mvKey_RShift)
		except Exception:
			try:
				shift_down = dpg.is_key_down(dpg.mvKey_Shift)
			except Exception:
				shift_down = False
		if self.enable_exclude_select and self.show_point_centers and shift_down:
			# initialize selection on first drag tick
			if not self._rect_select_active:
				# record start in drawing-local coordinates
				try:
					lx, ly = dpg.get_drawing_mouse_pos()
				except Exception:
					mx, my = dpg.get_mouse_pos()
					try:
						pmin = dpg.get_item_rect_min(self.draw_image_id)
					except Exception:
						pmin = dpg.get_item_rect_min(self.drawlist)
					lx = mx - pmin[0]
					ly = my - pmin[1]
				self._rect_select_start = (float(lx), float(ly))
				# include when ctrl is also down
				ctrl_down = False
				try:
					ctrl_down = dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl)
				except Exception:
					try:
						ctrl_down = dpg.is_key_down(dpg.mvKey_Control)
					except Exception:
						ctrl_down = False
				self._rect_select_include = bool(ctrl_down)
				self._rect_select_active = True
			# always update current end point
			try:
				lx, ly = dpg.get_drawing_mouse_pos()
			except Exception:
				mx, my = dpg.get_mouse_pos()
				try:
					pmin = dpg.get_item_rect_min(self.draw_image_id)
				except Exception:
					pmin = dpg.get_item_rect_min(self.drawlist)
				lx = mx - pmin[0]
				ly = my - pmin[1]
			self._rect_select_end = (float(lx), float(ly))
			# do not orbit while selecting
			return
		dx, dy = a[1], a[2]
		# compute incremental delta from cumulative drag deltas
		if self._left_drag_prev is None:
			self._left_drag_prev = (dx, dy)
			return
		inc_dx = dx - self._left_drag_prev[0]
		inc_dy = dy - self._left_drag_prev[1]
		self._left_drag_prev = (dx, dy)
		if abs(inc_dx) + abs(inc_dy) > 0:
			# use orbit_by so orbit follows rolled up/right
			self.cam.orbit_by(inc_dx, inc_dy, sensitivity=0.005)
			self._did_left_drag = True
			# sync sliders
			try:
				dpg.set_value(self.yaw_slider, self.cam.yaw)
				dpg.set_value(self.pitch_slider, self.cam.pitch)
			except Exception:
				pass
			self.needs_redraw = True

	def _on_mouse_drag_right(self, s, a):
		if not self._mouse_over_render():
			return
		dx, dy = a[1], a[2]
		# compute incremental delta from cumulative drag deltas
		if self._right_drag_prev is None:
			self._right_drag_prev = (dx, dy)
			return
		inc_dx = dx - self._right_drag_prev[0]
		inc_dy = dy - self._right_drag_prev[1]
		self._right_drag_prev = (dx, dy)
		if abs(inc_dx) + abs(inc_dy) > 0:
			self.cam.pan(inc_dx, inc_dy)
			self.needs_redraw = True

	def _on_mouse_wheel(self, s, a):
		if not self._mouse_over_render():
			return
		# a may be scalar delta or (x, y) where y is wheel delta
		if isinstance(a, (list, tuple)):
			if len(a) >= 2:
				delta = a[1]
			elif len(a) == 1:
				delta = a[0]
			else:
				delta = 0
		else:
			delta = a
		self.cam.radius = float(np.clip(self.cam.radius * (0.9 if delta > 0 else 1.1), 0.3, 5.0))
		self.needs_redraw = True

	def _on_screenshot(self):
		ts = time.strftime("%Y%m%d_%H%M%S")
		path = os.path.join(self.args.output_dir, f"shot_{ts}.npy")
		if hasattr(self, "last_frame"):
			np.save(path, self.last_frame)
			print("Saved", path)

	def _on_resize_height(self, s, a):
		# Adjust render size based on height slider, preserve aspect
		try:
			new_h = int(a)
		except Exception:
			return
		new_w = int(round(self.aspect * float(new_h)))
		if new_w == self.W and new_h == self.H:
			return
		self.W = new_w
		self.H = new_h
		# move controls to the right of the render area
		try:
			dpg.configure_item(self.render_win, width=self.W, height=self.H)
			dpg.configure_item(self.drawlist, width=self.W, height=self.H)
			dpg.configure_item(self.draw_image_id, pmin=(0, 0), pmax=(self.W, self.H))
			dpg.set_value(self.size_slider, self.H)
		except Exception:
			pass
		# update camera and texture
		# Camera stays at base resolution (do not change to avoid re-tracing rays count)
		self._recreate_texture_and_draw()

	def _on_render_window_resized(self, sender, app_data):
		# app_data is (new_width, new_height) but DearPyGui may not pass it consistently; query directly
		try:
			win_w = int(dpg.get_item_width(self.render_win))
			win_h = int(dpg.get_item_height(self.render_win))
		except Exception:
			return
		# Compute target size preserving aspect; shrink to fit if needed
		target_h = max(1, win_h)
		target_w = int(round(self.aspect * float(target_h)))
		if target_w > win_w:
			target_w = max(1, win_w)
			target_h = int(round(float(target_w) / max(1e-6, self.aspect)))
		# Update sizes and reposition the drawn image within drawlist
		if target_w != self.W or target_h != self.H:
			self.W = int(target_w)
			self.H = int(target_h)
			# Resize drawlist first (no texture change yet)
			try:
				dpg.configure_item(self.drawlist, width=self.W, height=self.H)
				dpg.configure_item(self.draw_image_id, pmin=(0, 0), pmax=(self.W, self.H))
				self._recreate_texture_and_draw()
			except Exception:
				self._recreate_texture_and_draw()

	def _on_select_view(self, s, a):
		# a may be label like "12 (IMG_0123.jpg)"; resolve to index
		idx = None
		if isinstance(a, str) and self.dataset_label_to_index:
			idx = self.dataset_label_to_index.get(a, None)
		if idx is None:
			try:
				idx = int(a)
			except Exception:
				idx = None
		self.dataset_cam_idx = idx
		if idx is not None:
			self._apply_dataset_pose_to_camera(idx)
		self.needs_redraw = True

	def _on_toggle_lock_view(self, s, a):
		# a is boolean
		self.use_dataset_pose = bool(a)
		self.needs_redraw = True

	def _on_toggle_overlay(self, s, a):
		self.show_point_centers = bool(a)
		self.needs_redraw = True

	def _on_toggle_exclude_mode(self, s, a):
		self.enable_exclude_select = bool(a)
		# enable/disable bulk buttons only when exclusion mode is on
		try:
			dpg.configure_item(self.btn_all_set, enabled=self.enable_exclude_select)
			dpg.configure_item(self.btn_all_reset, enabled=self.enable_exclude_select)
		except Exception:
			pass
		# when turning off, restore original scales so all MLPs are visible; keep state
		if not self.enable_exclude_select:
			self._restore_scene_scales()
		else:
			# turning on: apply current mask
			self._apply_exclusion_mask_to_scene()
		self.needs_redraw = True

	def _ensure_exclusion_state(self):
		# initialize exclusion structures if needed
		if self.excluded_mask is None:
			try:
				N = int(self.scene.positions.shape[0])
			except Exception:
				N = 0
			self.excluded_mask = torch.zeros(N, dtype=torch.bool)
			self._log_scales_backup = None

	def _apply_exclusion_mask_to_scene(self):
		# Apply the current excluded_mask by shrinking log_scales of excluded primitives
		self._ensure_exclusion_state()
		if self.excluded_mask is None or self.excluded_mask.numel() == 0:
			return
		try:
			device = self.scene.log_scales.device
			nls = self.scene.log_scales
			if self._log_scales_backup is None or self._log_scales_backup.shape != nls.shape:
				self._log_scales_backup = nls.detach().clone()
			new_vals = self._log_scales_backup.detach().clone().to(device)
			mask_dev = self.excluded_mask.to(device)
			if mask_dev.any():
				new_vals[mask_dev, :] = -20.0  # effectively zero radius
			with torch.no_grad():
				nls.copy_(new_vals)
		except Exception:
			pass
		self.needs_redraw = True

	def _restore_scene_scales(self):
		# Restore original log_scales from backup if available
		try:
			if self._log_scales_backup is not None and self.scene is not None:
				with torch.no_grad():
					self.scene.log_scales.copy_(self._log_scales_backup.to(self.scene.log_scales.device))
		except Exception:
			pass

	def _on_all_set(self):
		# Exclude all primitives (only active when exclusion mode is enabled)
		if not self.enable_exclude_select:
			return
		self._ensure_exclusion_state()
		if self.excluded_mask is not None and self.excluded_mask.numel() > 0:
			self.excluded_mask[:] = True
			self._apply_exclusion_mask_to_scene()

	def _on_all_reset(self):
		# Include all primitives back
		if not self.enable_exclude_select:
			return
		self._ensure_exclusion_state()
		if self.excluded_mask is not None and self.excluded_mask.numel() > 0:
			self.excluded_mask[:] = False
			self._apply_exclusion_mask_to_scene()

	def _on_select_ckpt(self, s, a):
		# a is the selected filename (basename)
		if not isinstance(a, str):
			return
		try:
			idx = self.ckpt_items.index(a)
		except Exception:
			return
		new_path = self.ckpt_paths[idx]
		if new_path == self.selected_ckpt_path:
			return
		self._reload_scene(new_path)
		self.needs_redraw = True

	def _on_roll_ccw(self, s=None, a=None):
		# CCW when looking forward (increase roll)
		self.cam.roll_by(+0.05)
		self.needs_redraw = True

	def _on_roll_cw(self, s=None, a=None):
		# CW when looking forward (decrease roll)
		self.cam.roll_by(-0.05)
		self.needs_redraw = True

	def _on_roll_released(self, s=None, a=None):
		# Do not rebase to global up here to preserve the effect of roll and avoid flicker
		# Subsequent orbit drags will rebase to global up on mouse release
		self.needs_redraw = True

	def _on_dolly_forward(self, s=None, a=None):
		# Move forward toward the scene along current forward direction
		step = 0.05 * float(self.cam.radius)
		self.cam.dolly(+step)
		self.needs_redraw = True

	def _on_dolly_backward(self, s=None, a=None):
		# Move backward away from the scene along current forward direction
		step = 0.05 * float(self.cam.radius)
		self.cam.dolly(-step)
		self.needs_redraw = True

	def _fov_from_focal(self, focal: float, H: int) -> float:
		# fov = 2 * atan((H/2)/focal)
		return math.degrees(2.0 * math.atan((0.5 * H) / max(1e-6, focal)))

	def _apply_dataset_pose_to_camera(self, idx: int):
		# Set orbit camera to match the dataset pose and FOV, but keep controls active
		if self.dataset is None:
			return
		try:
			pose = self.dataset.poses[idx]  # (3,4)
		except Exception:
			return
		cam_pos = pose[:3, 3].cpu().numpy()
		center = self.cam.center.cpu().numpy() if isinstance(self.cam.center, torch.Tensor) else np.array(self.cam.center)
		offset = cam_pos - center
		r = float(np.linalg.norm(offset) + 1e-8)
		dirv = offset / max(r, 1e-8)
		# yaw = atan2(x, z), pitch = asin(y)
		yaw = float(math.atan2(dirv[0], dirv[2]))
		pitch = float(np.clip(math.asin(dirv[1]), -1.2, 1.2))

		# update camera params
		self.cam.radius = r
		self.cam.yaw = yaw
		self.cam.pitch = pitch

		# match FOV to dataset focal scaled to viewer height
		if self.dataset_focal is not None and self.dataset_H is not None:
			focal_scaled = self.dataset_focal * (float(self.H) / float(self.dataset_H))
			fov_deg = self._fov_from_focal(focal_scaled, self.H)
			self.cam.fov_deg = float(np.clip(fov_deg, 30.0, 100.0))

		# update UI sliders to reflect new camera params
		try:
			dpg.set_value(self.yaw_slider, self.cam.yaw)
			dpg.set_value(self.pitch_slider, self.cam.pitch)
			dpg.set_value(self.radius_slider, self.cam.radius)
			dpg.set_value(self.fov_slider, self.cam.fov_deg)
		except Exception:
			pass

	def _generate_rays_from_dataset_pose(self):
		# Use selected dataset pose; generate rays at base resolution only
		pose = self.dataset.poses[self.dataset_cam_idx]  # (3,4)
		c2w = pose.to(self.device)
		Ht, Wt = self.base_H, self.base_W
		# scale focal by base height ratio (preserve vertical FOV)
		focal_scaled = self.dataset_focal * (float(Ht) / float(self.dataset_H))
		i, j = torch.meshgrid(torch.arange(Wt, device=self.device), torch.arange(Ht, device=self.device), indexing="xy")
		dirs = torch.stack([
			(i - Wt * 0.5) / focal_scaled,
			-(j - Ht * 0.5) / focal_scaled,
			torch.ones_like(i, dtype=torch.float32)
		], -1).to(torch.float32)
		rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1)
		rays_o = c2w[:3, 3].expand_as(rays_d)
		return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

	def _recreate_texture_and_draw(self):
		# Destroy and recreate dynamic texture and update draw_image size mapping
		try:
			# create a new texture first with current W,H
			new_tex = dpg.add_dynamic_texture(self.W, self.H, [0.0]*(self.W*self.H*4), parent=self.tex_reg)
			# update draw_image to map to new size
			if self.draw_image_id is not None:
				dpg.configure_item(self.draw_image_id, pmin=(0, 0), pmax=(self.W, self.H), texture_tag=new_tex)
			# resize drawlist as well
			if self.drawlist is not None:
				dpg.configure_item(self.drawlist, width=self.W, height=self.H)
			# swap to new texture; keep old texture alive to avoid potential segfaults
			self.tex_id = new_tex
		except Exception as e:
			print("[viewer] Failed to recreate texture:", e)
		# Keep camera at base resolution; aspect remains constant from initialization
		self.needs_redraw = True

	def _project_points_to_pixels(self, points_world: torch.Tensor, c2w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""
		Project 3D points to pixel coordinates for current camera.
		Returns (px, py, valid_mask) in tensor on CPU.
		"""
		w2c_R = c2w[:3, :3].T
		w2c_t = -w2c_R @ c2w[:3, 3]
		pts_cam = (w2c_R @ points_world.T) + w2c_t.unsqueeze(1)  # (3,N)
		x = pts_cam[0]
		y = pts_cam[1]
		z = pts_cam[2]
		valid = z > 1e-6
		# compute focal for current display size (preserve camera FOV)
		fov = math.radians(self.cam.fov_deg)
		f_disp = 0.5 * float(self.H) / math.tan(0.5 * fov)
		px = (x * f_disp / z) + (self.W * 0.5)
		# flip Y to match ray generation convention (image Y down)
		py = (-y * f_disp / z) + (self.H * 0.5)
		return px.cpu(), py.cpu(), valid.cpu()

	def _draw_point_centers_overlay(self, frame_cpu: np.ndarray, c2w: torch.Tensor) -> np.ndarray:
		# frame_cpu: (H,W,4) float32 in [0,1]
		if not self.show_point_centers:
			return frame_cpu
		with torch.no_grad():
			points = self.scene.positions  # (N,3) on device
			px, py, valid = self._project_points_to_pixels(points, c2w)
			px = px.numpy()
			py = py.numpy()
			valid = valid.numpy()
			h, w = self.H, self.W
			r = int(max(1, self.overlay_point_radius_px))
			mask_np = None
			try:
				if self.excluded_mask is not None and self.excluded_mask.numel() == points.shape[0]:
					mask_np = self.excluded_mask.numpy()
			except Exception:
				mask_np = None
			# draw points: yellow by default, red if excluded
			for i in range(px.shape[0]):
				if not valid[i]:
					continue
				xi = int(round(px[i]))
				yi = int(round(py[i]))
				if xi < 0 or xi >= w or yi < 0 or yi >= h:
					continue
				is_excl = False
				if mask_np is not None and i < len(mask_np):
					is_excl = bool(mask_np[i])
				# small disk
				for dy in range(-r, r+1):
					for dx in range(-r, r+1):
						if dx*dx + dy*dy <= r*r:
							xp = xi + dx
							yp = yi + dy
							if 0 <= xp < w and 0 <= yp < h:
								if is_excl:
									frame_cpu[yp, xp, 0] = 1.0
									frame_cpu[yp, xp, 1] = 0.0
									frame_cpu[yp, xp, 2] = 0.0
								else:
									frame_cpu[yp, xp, 0] = 1.0
									frame_cpu[yp, xp, 1] = 1.0
									frame_cpu[yp, xp, 2] = 0.0
		return frame_cpu

	def _current_c2w_for_projection(self) -> torch.Tensor:
		# build the c2w used for current rendering path, including dataset normalization
		if self.use_dataset_pose and (self.dataset is not None) and (self.dataset_cam_idx is not None):
			c2w_used = self.dataset.poses[self.dataset_cam_idx].to(self.device)
		else:
			c2w_used = self.cam.get_pose().to(self.device)
			# map to training coords if dataset normalization exists
			if self.dataset is not None and hasattr(self.dataset, 'avg_position') and hasattr(self.dataset, 'scale'):
				try:
					avg_np = self.dataset.avg_position
					scale_val = float(self.dataset.scale)
					avg = torch.tensor(avg_np, dtype=torch.float32, device=self.device)
					c2w_used = c2w_used.clone()
					c2w_used[:3, 3] = (c2w_used[:3, 3] - avg) * scale_val
				except Exception:
					pass
		return c2w_used

	def _on_mouse_click(self, s, a):
		# Left-click handler; perform exclusion toggle when enabled and overlay visible with Ctrl pressed
		try:
			if not self.enable_exclude_select or not self.show_point_centers:
				return
			# check ctrl key
			ctrl_down = False
			try:
				ctrl_down = dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl)
			except Exception:
				try:
					ctrl_down = dpg.is_key_down(dpg.mvKey_Control)
				except Exception:
					ctrl_down = False
			if not ctrl_down:
				return
			if not self._mouse_over_render():
				return
			# mouse position in drawing local coords
			try:
				lx, ly = dpg.get_drawing_mouse_pos()
			except Exception:
				mx, my = dpg.get_mouse_pos()
				# prefer the image's rect if available, fallback to drawlist
				try:
					pmin = dpg.get_item_rect_min(self.draw_image_id)
				except Exception:
					pmin = dpg.get_item_rect_min(self.drawlist)
				lx = mx - pmin[0]
				ly = my - pmin[1]
			if lx < 0 or ly < 0 or lx >= self.W or ly >= self.H:
				return
			self._ensure_exclusion_state()
			if self.excluded_mask is None or self.excluded_mask.numel() == 0:
				return
			c2w_used = self._current_c2w_for_projection()
			with torch.no_grad():
				points = self.scene.positions
				px, py, valid = self._project_points_to_pixels(points, c2w_used)
				px = px.numpy(); py = py.numpy(); valid = valid.numpy()
				best_i = -1; best_d2 = 1e9
				pick_r = max(3, int(self.overlay_point_radius_px) + 2)
				for i in range(px.shape[0]):
					if not valid[i]:
						continue
					dx = px[i] - lx; dy = py[i] - ly
					d2 = float(dx*dx + dy*dy)
					if d2 < best_d2 and d2 <= (pick_r * pick_r):
						best_d2 = d2; best_i = i
				if best_i >= 0:
					self.excluded_mask[best_i] = ~self.excluded_mask[best_i]
					self._apply_exclusion_mask_to_scene()
					self.needs_redraw = True
		except Exception:
			pass

	def _render_once(self):
		# 1) Render base resolution with MLP
		if self.use_dataset_pose and (self.dataset is not None) and (self.dataset_cam_idx is not None):
			rays_o, rays_d = self._generate_rays_from_dataset_pose()
			Ht_exp, Wt_exp = self.base_H, self.base_W
			c2w_used = self.dataset.poses[self.dataset_cam_idx].to(self.device)
		else:
			# camera rays from interactive orbit camera
			rays_o, rays_d = self.cam.generate_rays(self.device)
			Ht_exp, Wt_exp = int(self.cam.height), int(self.cam.width)
			c2w_used = self.cam.get_pose().to(self.device)
			# If dataset normalization exists, map camera pose and rays to training (normalized) coords
			if self.dataset is not None and hasattr(self.dataset, 'avg_position') and hasattr(self.dataset, 'scale'):
				try:
					avg_np = self.dataset.avg_position
					scale_val = float(self.dataset.scale)
					avg = torch.tensor(avg_np, dtype=torch.float32, device=self.device)
					# normalize ray origins: o' = scale * (o - avg)
					rays_o = (rays_o - avg) * scale_val
					# directions need no scaling (unit-length downstream) so keep as is
					# normalize camera pose translation similarly
					c2w_used = c2w_used.clone()
					c2w_used[:3, 3] = (c2w_used[:3, 3] - avg) * scale_val
				except Exception:
					pass
		# guard against any mismatch between expected HxW and rays count
		num_rays = int(rays_o.shape[0])
		if Ht_exp * Wt_exp != num_rays:
			# try base dims
			if self.base_H * self.base_W == num_rays:
				Ht_exp, Wt_exp = self.base_H, self.base_W
			# try camera dims
			elif int(self.cam.height) * int(self.cam.width) == num_rays:
				Ht_exp, Wt_exp = int(self.cam.height), int(self.cam.width)
			# try factoring by base_W
			elif self.base_W > 0 and num_rays % self.base_W == 0:
				Ht_exp, Wt_exp = num_rays // self.base_W, self.base_W
			# try factoring by base_H
			elif self.base_H > 0 and num_rays % self.base_H == 0:
				Ht_exp, Wt_exp = self.base_H, num_rays // self.base_H
		# render at inferred H,W
		rgb = render_scene(self.scene, rays_o, rays_d, Ht_exp, Wt_exp, batch_size=self.batch_size)
		# if inferred dims differ from base dims, adapt base frame accordingly
		if (Ht_exp, Wt_exp) != (self.base_H, self.base_W):
			frame_base = torch.cat([rgb, torch.ones(Ht_exp, Wt_exp, 1, device=rgb.device)], dim=-1)
			# resample to base for consistent caching
			fb = frame_base.permute(2, 0, 1).unsqueeze(0)
			fb_base = F.interpolate(fb, size=(self.base_H, self.base_W), mode='bilinear', align_corners=False)
			frame_base = fb_base.squeeze(0).permute(1, 2, 0)
		else:
			frame_base = torch.cat([rgb, torch.ones(self.base_H, self.base_W, 1, device=rgb.device)], dim=-1)
		# 2) Upsample to current display size with bilinear (no extra dependency)
		if self.W != self.base_W or self.H != self.base_H:
			fb = frame_base.permute(2, 0, 1).unsqueeze(0)  # 1x4xHb x Wb
			fb_up = F.interpolate(fb, size=(self.H, self.W), mode='bilinear', align_corners=False)
			frame_disp = fb_up.squeeze(0).permute(1, 2, 0)
		else:
			frame_disp = frame_base
		frame_disp_cpu = frame_disp.detach().cpu().numpy().astype(np.float32)
		self.last_frame_base = frame_base.detach().cpu().numpy().astype(np.float32)
		# 3) Optional overlay in display space
		frame_disp_cpu = self._draw_point_centers_overlay(frame_disp_cpu, c2w_used)
		# 4) Upload to texture
		dpg.set_value(self.tex_id, frame_disp_cpu.reshape(-1).tolist())
		self.last_frame = frame_disp_cpu

	def _scan_ckpt_dir(self):
		# Populate ckpt_items and ckpt_paths sorted by mtime desc
		files = []
		for name in os.listdir(self.ckpt_dir):
			if name.endswith('.pt'):
				full = os.path.join(self.ckpt_dir, name)
				try:
					mtime = os.path.getmtime(full)
				except Exception:
					mtime = 0.0
				files.append((mtime, name, full))
		files.sort(key=lambda x: x[0], reverse=True)
		self.ckpt_items = [n for _, n, _ in files]
		self.ckpt_paths = [p for _, _, p in files]

	def _reload_scene(self, ckpt_path: str):
		# Replace current scene with a new one from ckpt_path
		try:
			new_scene = build_scene_from_ckpt(ckpt_path, self.device)
		except Exception as e:
			print(f"[viewer] Failed to load checkpoint {ckpt_path}: {e}")
			return
		# swap
		old = getattr(self, 'scene', None)
		self.scene = new_scene
		self.selected_ckpt_path = ckpt_path
		# reset exclusion state for the new scene
		try:
			N = int(self.scene.positions.shape[0])
			self.excluded_mask = torch.zeros(N, dtype=torch.bool)
			self._log_scales_backup = None
		except Exception:
			self.excluded_mask = None
			self._log_scales_backup = None
		# best effort free
		try:
			del old
			if self.device.type == 'cuda':
				torch.cuda.empty_cache()
		except Exception:
			pass

	def run(self):
		while dpg.is_dearpygui_running():
			now = time.time()
			# aim ~3-5 FPS when interacting, otherwise show last frame
			if self.needs_redraw or (now - self.last_time) > (1.0 / 3.0):
				self._render_once()
				self.last_time = now
				self.needs_redraw = False
			dpg.render_dearpygui_frame()

		dpg.destroy_context()


def main():
	args = parse_args()
	app = ViewerApp(args)
	app.run()


if __name__ == "__main__":
	main()


