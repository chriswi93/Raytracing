"""
RaytracingEngine sample class.
"""
import numpy as np
import cv2
from typing import Tuple
from .geometry import Linear3DTransformation, Line3D, Screen3DPlane

class RaytracingEngine:
    """
    Class renders frames of vertices in three dimensional space and draws surface, grid and keypoints.
    """
    def __init__(self,
                 sensor_width: int = 1024,
                 sensor_height: int = 800,
                 fov_degrees: int = 77,
                 focal_length_pu: int = 26,
                 bg_color: Tuple[int] = (0,0,0),
                 keypoints_color: Tuple[int] = (0,0,255),
                 keypoints_size: int = 10) -> None:
        """Create new RaytracingEngine instance.

        Args:
            sensor_width (int, optional): sensor width in pixels. Defaults to 1024.
            sensor_height (int, optional): sensor height in pixels. Defaults to 800.
            fov_degrees (int, optional): field of view in degrees. Defaults to 77.
            focal_length_pu (int, optional): focal length in physical units. Defaults to 26.
            bg_color (Tuple[int], optional): background color (rgb). Defaults to (0,0,0).
            keypoints_color (Tuple[int], optional): keypoints color (rgb). Defaults to (0,0,255).
            keypoints_size (int, optional): keypoints size. Defaults to 10.
        """
        self.bg_color = bg_color
        self.keypoints_color = keypoints_color
        self.keypoints_size = keypoints_size

        # initialize view and camera
        self.z, self.ppu = self.init_view((sensor_height, sensor_width), fov_degrees, focal_length_pu)
        self.init_camera((0,0, -self.z))

    def _trace_ray_screen2d(self, vertices3d: np.ndarray) -> np.ndarray:
        """Returns 2d screen coordinates for each provided 3d vertex position.

        Args:
            vertices3d (np.ndarray): 3d vertex positions

        Returns:
            np.ndarray: 2d screen coordinates
        """
        # compute intersection points
        intersections = [self.screen_3dplane.intersects(Line3D(self.eye_xyz, vertex)) for vertex in vertices3d]
        vertices2d = np.float32(intersections)[:,:2] # drop third dimension
        return vertices2d
        
    def _adjust_out_of_screen(self, vertices2d: np.ndarray) -> None:
        """Adjusts the position of all 2d vertices if at least one vertex is not visible.

        Args:
            vertices2d (np.ndarray): 2d vertices on screen
        """
        vertices2d_adjust_h = vertices2d[vertices2d[:,0] < 0, 0]
        vertices2d_adjust_w = vertices2d[vertices2d[:,1] < 0, 1]
        if len(vertices2d_adjust_h) > 0:
            vertices2d[:,0] = vertices2d[:,0] - vertices2d_adjust_h.min()
        if len(vertices2d_adjust_w) > 0:
            vertices2d[:,1] = vertices2d[:,1] - vertices2d_adjust_w.min()
            
        vertices2d_adjust_h = vertices2d[vertices2d[:,0] > self.w, 0] - self.w
        vertices2d_adjust_w = vertices2d[vertices2d[:,1] > self.h, 1] - self.h
        if len(vertices2d_adjust_h) > 0:
            vertices2d[:,0] = vertices2d[:,0] - vertices2d_adjust_h.max()
        if len(vertices2d_adjust_w) > 0:
            vertices2d[:,1] = vertices2d[:,1] - vertices2d_adjust_w.max()
    
    def _transform_vertices3d(self,
                              vertices3d: np.ndarray,
                              rotate_xyz: np.ndarray,
                              scale_xyz: float,
                              translate_xyz: np.ndarray) -> np.ndarray:
        """Applies linear 3d transformations to given vertex positions and returns transformed vertex positions.

        Args:
            vertices3d (np.ndarray): 3d vertices
            rotate_xyz (np.ndarray): rotation for all axis
            scale_xyz (float): scale factor
            translate_xyz (np.ndarray): translation for all axis

        Returns:
            np.ndarray: transformed vertex positions
        """
        vertices3d = Linear3DTransformation.rotate(vertices3d, rotate_xyz)
        vertices3d = Linear3DTransformation.scale(vertices3d, scale_xyz)
        vertices3d = Linear3DTransformation.translate(vertices3d, translate_xyz)
        return vertices3d

    def render_frame(self,
                     rotate_xyz: np.ndarray,
                     translate_xyz: np.ndarray,
                     scale_xyz: float,
                     draw_surface: bool = True,
                     draw_grid: bool = True,
                     draw_keypoints: bool = True,
                     adjust_out_of_screen: bool = False) -> np.ndarray:
        """Renders and returns a frame.

        Args:
            rotate_xyz (np.ndarray): rotation for all axis
            translate_xyz (np.ndarray): translation for all axis
            scale_xyz (float): scale factor
            draw_surface (bool, optional): True to draw a surface. Defaults to True.
            draw_grid (bool, optional): True to draw a grid. Defaults to True.
            draw_keypoints (bool, optional): True to draw keypoints. Defaults to True.
            adjust_out_of_screen (bool, optional): True to auto adjust positions if some vertices are not visible. Defaults to False.

        Returns:
            np.ndarray: rendered frame
        """
        # apply linear 3d transformations to vertices and store 3d positions
        self.vertices3d = self._transform_vertices3d(self.vertices, rotate_xyz, scale_xyz, translate_xyz)

        # project 3d vertices on screen and get 2d pixel positions on screen
        vertices2d = self._trace_ray_screen2d(self.vertices3d)
        # shift the coordinate origin to the center of the screen
        vertices2d = vertices2d + self.vertices2d_correction
        
        # auto adjust positions in place
        if adjust_out_of_screen is True:
            self._adjust_out_of_screen(vertices2d)
        
        # store 2d positions and draw surface, grid and keypoints
        self.vertices2d = vertices2d
        frame = self._draw(vertices2d, draw_surface, draw_grid, draw_keypoints)
        return frame
    
    def _draw(self, vertices2d: np.ndarray, draw_surface: bool, draw_grid: bool, draw_keypoints: bool) -> np.ndarray:
        """Draws surface, grid and keypoints on frame.

        Args:
            vertices2d (np.ndarray): vertex positions on screen.
            draw_surface (bool): True to draw a surface.
            draw_grid (bool): True to draw a grid.
            draw_keypoints (bool): True to draw keypoints.

        Returns:
            np.ndarray: frame
        """
        if draw_surface is True:
            frame = self._draw_surface(vertices2d[:4])
        elif draw_grid is True:
            frame = self._draw_grid(vertices2d[:4])
        else:
            frame = self.sensor2d
            frame[...] = self.bg_color
        if draw_keypoints is True:
            frame = self._draw_keypoints(frame, vertices2d)
        return frame

    def _draw_keypoints(self, frame: np.ndarray, vertices2d: np.ndarray) -> np.ndarray:
        """Draws keypoints on frame.

        Args:
            frame (np.ndarray): frame
            vertices2d (np.ndarray): vertex positions on screen

        Returns:
            np.ndarray: frame
        """
        kp = [cv2.KeyPoint(i[0], i[1], size=self.keypoints_size) for i in vertices2d]
        return cv2.drawKeypoints(frame.astype(np.uint8), kp, None, color=self.keypoints_color, flags=4)

    def _draw_grid(self, vertices2d: np.ndarray) -> np.ndarray:
        """Draws grid lines on frame.

        Args:
            vertices2d (np.ndarray): vertex positions on screen

        Returns:
            np.ndarray: frame
        """
        return cv2.polylines(self.sensor2d, [np.int32(vertices2d)], isClosed=True, color=(255,255,255), thickness=2, lineType=cv2.LINE_AA)

    def _draw_surface(self, vertices2d: np.ndarray) -> np.ndarray:
        """Draws surface on frame.

        Args:
            vertices2d (np.ndarray): vertex positions on screen

        Returns:
            np.ndarray: frame
        """
        M = cv2.getPerspectiveTransform(self.img_2d_coords, np.float32(vertices2d), cv2.INTER_LANCZOS4)
        return cv2.warpPerspective(self.img, M, (self.w, self.h), borderValue=self.bg_color)
    
    def _get_focal_length(self, w: int, h: int, fov_degrees: int) -> float:
        """Returns focal length.

        Args:
            w (int): screen width in pixels
            h (int): screen height in pixels
            fov_degrees (int): field of view in dregrees

        Returns:
            float: focal length
        """
        angle = np.deg2rad(fov_degrees)
        diag = np.sqrt(h**2 + w**2)
        a = diag / (2 * np.sin(angle / 2))
        z = 0.5 * np.sqrt(4 * a * a - diag * diag)
        return z

    def init_view(self, sensor_size: Tuple[int], fov_degrees: int, focal_len: int) -> Tuple[float, float]:
        """Initializes a new camera viewpoint.

        Args:
            sensor_size (Tuple[int]): sensor size in pixels (height, width)
            fov_degrees (int): field of view in degrees
            focal_len (int): focal length

        Returns:
            Tuple[float, float]: camera viewpoint position on z-axis, pixel per unit (ppu)
        """
        h, w = np.int32(sensor_size)
        if getattr(self, "h", -1) != h or getattr(self, "w", -1) != w:
            self.sensor2d = np.zeros((h, w, 3), dtype=np.uint8)
            self.h, self.w = h, w
        
        self.screen_3dplane = Screen3DPlane(w, h)
        z = self._get_focal_length(w, h, fov_degrees)
        self.vertices2d_correction = np.array([w, h]) / 2
        return z, z / focal_len

    def init_camera(self, pos: Tuple[float, float, float]) -> None:
        """Initializes a new camera position.

        Args:
            pos (Tuple[float, float, float]): xyz camera position
        """
        self.eye_xyz = np.float64(pos)

    def set_image(self, img: np.ndarray, img_height_pu: int, surface_downsampling: bool = True) -> None:
        """Sets a new image.

        Args:
            img (np.ndarray): image
            img_height_pu (int): image height in physical units
            surface_downsampling (bool, optional): Surface downsampling improves frame rendering performance. Defaults to True.
        """
        print("Image shape: {}".format(img.shape))
        if surface_downsampling is True:
            height, _ = img.shape[:2]
            scale = self.h / height
            new_x, new_y = img.shape[1] * scale, img.shape[0] * scale
            img = cv2.resize(img, (int(new_x), int(new_y)), cv2.INTER_AREA)
            print("Image shape after downsampling: {}".format(img.shape))
        self.img_height_pu = img_height_pu
        self.img = img
        self.default_scale_factor = img_height_pu / self.pixel2pu(img.shape[0])
        self.img_2d_coords = np.array(((img.shape[1], 0), (0, 0), (0, img.shape[0]), (img.shape[1], img.shape[0])), dtype=np.float32)
        self.vertices = np.array([
            [-img.shape[1]/2, img.shape[0]/2, 0],
            [img.shape[1]/2, img.shape[0]/2, 0],
            [img.shape[1]/2, -img.shape[0]/2, 0],
            [-img.shape[1]/2, -img.shape[0]/2, 0]
        ])

    def get_default_scale_factor(self):
        """Returns the default scale factor. The default scale factor is equal to the size of the object in physical units.

        Returns:
            float: default scale factor
        """
        return self.default_scale_factor
    
    def pu2pixel(self, values: np.ndarray) -> np.ndarray:
        """Returns the pixel values for physical units.

        Args:
            values (np.ndarray): values in physical units

        Returns:
            np.ndarray: values in pixels
        """
        return self.ppu * values
    
    def pixel2pu(self, values: np.ndarray) -> np.ndarray:
        """Returns physical units for pixel values.

        Args:
            values (np.ndarray): values in physical units

        Returns:
            np.ndarray: values in physical units
        """
        return values / self.ppu
    
    def add_vertices(self, vertices: np.ndarray) -> None:
        """Adds new vertex positions to render.

        Args:
            vertices (np.ndarray): vertex positions
        """
        self.vertices = np.vstack([self.vertices, vertices])
