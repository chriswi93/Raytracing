"""
Helper classes to simplify transformations and calculations in 3d space.
Implementation is optimized for speed (faster compared to e.g. SymPy).
"""
import pyquaternion as pq
import numpy as np

class Linear3DTransformation:
    """
    Linear3DTransformation class provides access to rotation, translation and scaling in 3d space.
    """

    @staticmethod
    def rotate(vertices: np.ndarray, rotate_xyz: np.ndarray) -> np.ndarray:
        """Returns rotated vertex positions in 3d space.

        Args:
            vertices (np.ndarray): vertex positions
            rotate_xyz (np.ndarray): xyz axis rotation in degress

        Returns:
            np.ndarray: rotated vertex positions
        """
        qt_x = pq.Quaternion(axis=[1, 0, 0], degrees=rotate_xyz[0])
        qt_y = pq.Quaternion(axis=[0, 1, 0], degrees=rotate_xyz[1])
        qt_z = pq.Quaternion(axis=[0, 0, 1], degrees=rotate_xyz[2])
        qt = qt_x * qt_y * qt_z
        return vertices.dot(qt.rotation_matrix.T)
    
    @staticmethod
    def translate(vertices: np.ndarray, shift: np.ndarray) -> np.ndarray:
        """Returns shifted vertex positions in 3d space.

        Args:
            vertices (np.ndarray): vertex positions
            shift (np.ndarray): xyz axis shift

        Returns:
            np.ndarray: shifted vertex positions
        """
        return vertices + shift
    
    @staticmethod
    def scale(vertices: np.ndarray, factor: np.ndarray) -> np.ndarray:
        """Returns scaled vertex positions in 3d space.

        Args:
            vertices (np.ndarray): vertex positions
            factor (np.ndarray): xyz axis shift

        Returns:
            np.ndarray: scaled vertex positions
        """
        return vertices * factor

class Line3D:
    def __init__(self, p1: np.ndarray, p2: np.ndarray):
        """Line3D class represents a line in 3d space.

        Args:
            p1 (np.ndarray): _description_
            p2 (np.ndarray): _description_
        """
        self.support_vector = p1
        self.direction_vector = p2 - p1
    
    def __call__(self, lambda_value: float) -> np.ndarray:
        """Returns any point on the 3d line given lambda.

        Args:
            lambda_value (float): lambda value

        Returns:
            np.ndarray: xyz position
        """
        return self.support_vector + lambda_value * self.direction_vector

class Screen3DPlane:

    def __init__(self, width: int, height: int):
        """Screen3DPlane class represents a plane in 3d space.

        Args:
            width (int): plane width
            height (int): plane height
        """
        self.support_vector = self._support_vector_plane(width, height)
        self.normal_vector = self._normal_vector_plane(width, height, self.support_vector)

    def _normal_vector_plane(self, width: int, height: int, support_vector: np.ndarray) -> np.ndarray:
        """Returns normal vector for plane.

        Args:
            width (int): plane width
            height (int): plane height
            support_vector (np.ndarray): plane support vector

        Returns:
            np.ndarray: normal vector for plane
        """
        B = np.float64([-width/2, -height/2, 0])
        C = np.float64([width/2, height/2, 0])
        return np.cross(B - support_vector, C - support_vector)
        
    def _support_vector_plane(self, width: int, height: int) -> np.ndarray:
        """Returns support vector for plane.

        Args:
            width (int): plane width
            height (int): plane height

        Returns:
            np.ndarray: support vector for plane
        """
        return np.float64([-width/2, height/2, 0])
    
    def intersects(self, line: Line3D) -> np.ndarray:
        """Returns intersection point with another line.

        Args:
            line (Line3D): another line

        Returns:
            np.ndarray: xyz intersection point
        """
        lambda_value = (self.normal_vector.dot(self.support_vector) - self.normal_vector.dot(line.support_vector)) / self.normal_vector.dot(line.direction_vector)
        return line(lambda_value)
