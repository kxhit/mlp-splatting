import math
import torch


class OrbitCamera:
    def __init__(self, width: int, height: int, fov_deg: float = 60.0):
        self.width = width
        self.height = height
        self.fov_deg = fov_deg
        self.radius = 2.0
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self.center = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        # Control up vector defines yaw axis and pitch right via cross(control_up, forward)
        # This is updated on interaction end to "commit" the current pose as reference.
        self.control_up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
        self.global_up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)

    @property
    def focal(self) -> float:
        # pinhole focal from vertical FOV
        fov = math.radians(self.fov_deg)
        return 0.5 * self.height / math.tan(0.5 * fov)

    def get_pose(self) -> torch.Tensor:
        # camera position in world using spherical coordinates
        cp = math.cos(self.pitch)
        sp = math.sin(self.pitch)
        cy = math.cos(self.yaw)
        sy = math.sin(self.yaw)
        # forward vector from center to camera
        cam_dir = torch.tensor([cp * sy, sp, cp * cy], dtype=torch.float32)
        cam_pos = self.center + cam_dir * self.radius

        # build camera-to-world (right-handed, z forward) using committed control_up
        forward = (self.center - cam_pos)
        forward = forward / (forward.norm() + 1e-8)
        up_hint = self.control_up
        # if control_up is nearly parallel to forward, fall back to global up
        if torch.abs(torch.dot(up_hint, forward)) > 0.99:
            up_hint = self.global_up
        right = torch.cross(up_hint, forward, dim=0)
        right = right / (right.norm() + 1e-8)
        up = torch.cross(forward, right, dim=0)

        c2w = torch.zeros(3, 4, dtype=torch.float32)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = forward
        c2w[:3, 3] = cam_pos
        return c2w
    
    def set_pose(self, c2w: torch.Tensor) -> None:
        """
        Set yaw, pitch, center, and control_up from a given camera-to-world pose.
        - Uses existing self.radius for the orbit radius.
        - Assumes c2w columns: [right, up, forward, cam_pos], z-forward (camera +Z looks at scene).
        """
        # accept 3x4 or 4x4
        if c2w.shape[-2:] == (4, 4):
            c2w = c2w[:3, :4]
        elif c2w.shape[-2:] != (3, 4):
            raise ValueError(f"c2w must be (3,4) or (4,4), got {tuple(c2w.shape)}")

        c2w = c2w.to(dtype=torch.float32)

        right = c2w[:3, 0]
        up = c2w[:3, 1]
        forward = c2w[:3, 2]
        cam_pos = c2w[:3, 3]

        # normalize basis just in case
        eps = 1e-8
        forward = forward / (forward.norm() + eps)
        up = up / (up.norm() + eps)
        right = right / (right.norm() + eps)

        # center lies along camera forward at distance = self.radius
        center = cam_pos + forward * float(self.radius)

        # vector from center to camera defines spherical (yaw, pitch)
        cam_dir = cam_pos - center
        cam_dir = cam_dir / (cam_dir.norm() + eps)

        x = float(cam_dir[0])
        y = float(cam_dir[1])
        z = float(cam_dir[2])

        # graphics convention used in this class:
        # cam_dir = [cos(pitch)*sin(yaw), sin(pitch), cos(pitch)*cos(yaw)]
        new_pitch = math.asin(y)
        new_yaw = math.atan2(x, z)

        self.center = center
        self.pitch = new_pitch
        self.yaw = new_yaw

        # commit control frame's up from given pose
        self.control_up = up / (up.norm() + eps)

    def generate_rays(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        H, W = self.height, self.width
        i, j = torch.meshgrid(torch.arange(W, device=device), torch.arange(H, device=device), indexing="xy")
        dirs = torch.stack([
            (i - W * 0.5) / self.focal,
            (j - H * 0.5) / self.focal,
            torch.ones_like(i, dtype=torch.float32)
        ], -1).to(torch.float32)  # (H,W,3)

        c2w = self.get_pose().to(device)
        rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1)
        rays_o = c2w[:3, 3].expand_as(rays_d)
        return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    def pan(self, dx_pixels: float, dy_pixels: float) -> None:
        """
        Translate the camera center on the plane perpendicular to the viewing direction
        based on mouse drag in pixels. Positive dx is mouse right, positive dy is mouse down.
        """
        # world distance per pixel at the target depth (approximately the orbit radius)
        fov = math.radians(self.fov_deg)
        world_per_pixel = 2.0 * self.radius * math.tan(0.5 * fov) / max(1.0, float(self.height))

        # use current camera basis vectors
        c2w = self.get_pose()
        right = c2w[:3, 0]
        up = c2w[:3, 1]

        # dragging right should move the scene right (so center moves left -> -dx * right)
        # dragging down should move the scene down (so center moves +dy * up)
        delta = (-float(dx_pixels) * right + float(dy_pixels) * up) * float(world_per_pixel)
        self.center = self.center + delta

    def roll_by(self, delta_rad: float) -> None:
        # Rotate control_up around the current viewing direction to change the control frame,
        # without visually rolling the image (render roll remains 0 during orbiting).
        # Compute forward from spherical yaw/pitch
        cp = math.cos(self.pitch)
        sp = math.sin(self.pitch)
        cy = math.cos(self.yaw)
        sy = math.sin(self.yaw)
        cam_dir = torch.tensor([cp * sy, sp, cp * cy], dtype=torch.float32)
        forward = -(cam_dir / (cam_dir.norm() + 1e-8))

        def rotate_vec(vec: torch.Tensor, axis: torch.Tensor, angle: float) -> torch.Tensor:
            axis_n = axis / (axis.norm() + 1e-8)
            c = math.cos(angle)
            s = math.sin(angle)
            return vec * c + torch.linalg.cross(axis_n, vec) * s + axis_n * torch.dot(axis_n, vec) * (1.0 - c)

        self.control_up = rotate_vec(self.control_up, forward, float(delta_rad))
        self.control_up = self.control_up / (self.control_up.norm() + 1e-8)

    def orbit_by(self, dx_pixels: float, dy_pixels: float, sensitivity: float = 0.005) -> None:
        """
        Orbit the camera around the center using a stable world-up frame.
        - Horizontal drag (dx) performs yaw around world up (0,1,0)
        - Vertical drag (dy) performs pitch around the camera-right defined by world up × forward
        Roll does not affect the control axes so yaw/pitch remain decoupled.
        """
        # derive current camera position from yaw/pitch directly (render roll is not used for control axes)
        cp = math.cos(self.pitch)
        sp = math.sin(self.pitch)
        cy = math.cos(self.yaw)
        sy = math.sin(self.yaw)
        cam_dir = torch.tensor([cp * sy, sp, cp * cy], dtype=torch.float32)
        cam_pos = self.center + cam_dir * self.radius
        center = self.center
        v = (cam_pos - center)  # from center to camera

        def rotate_vec(vec: torch.Tensor, axis: torch.Tensor, angle: float) -> torch.Tensor:
            axis_n = axis / (axis.norm() + 1e-8)
            c = math.cos(angle)
            s = math.sin(angle)
            return vec * c + torch.linalg.cross(axis_n, vec) * s + axis_n * torch.dot(axis_n, vec) * (1.0 - c)

        # build axes from committed control up (which may be tilted)
        control_up = self.control_up
        forward = (center - cam_pos)
        forward = forward / (forward.norm() + 1e-8)
        right_base = torch.linalg.cross(control_up, forward)
        right_base = right_base / (right_base.norm() + 1e-8)

        # yaw: rotate around control up by -dx (drag right -> orbit camera right)
        ang_yaw = -float(dx_pixels) * sensitivity
        v = rotate_vec(v, control_up, ang_yaw)
        # pitch: rotate around right_base by -dy (drag down -> rotate downward)
        ang_pitch = -float(dy_pixels) * sensitivity
        v = rotate_vec(v, right_base, ang_pitch)

        # recompute spherical yaw/pitch from new cam_dir
        cam_dir = v / (v.norm() + 1e-8)
        y = float(cam_dir[1])
        x = float(cam_dir[0])
        z = float(cam_dir[2])
        new_pitch = float(math.asin(y))
        new_yaw = float(math.atan2(x, z))
        self.pitch = new_pitch
        self.yaw = new_yaw

    def commit_up_from_camera(self) -> None:
        """Set control_up to the camera's current up direction (from pose)."""
        c2w = self.get_pose()
        self.control_up = c2w[:3, 1]
        self.control_up = self.control_up / (self.control_up.norm() + 1e-8)

    def dolly(self, distance: float) -> None:
        """
        Move the camera forward/backward along its current forward direction by translating
        both the camera and the scene center together. For an orbit camera, this is achieved
        by shifting the center along the forward vector; the camera position follows
        implicitly since it is defined relative to the center.
        """
        c2w = self.get_pose()
        forward = c2w[:3, 2]
        self.center = self.center + forward * float(distance)

