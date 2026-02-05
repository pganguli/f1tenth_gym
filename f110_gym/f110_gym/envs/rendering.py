"""
Rendering engine for f1tenth gym env based on pyglet and OpenGL
Author: Hongrui Zheng
Updated for pyglet 2.x compatibility
"""

# opengl stuff
import pyglet

# In pyglet 2.x, legacy GL functions need to be imported from gl_compat or use ctypes
# We use pyglet's built-in projection handling instead
from pyglet.gl import Config

# other
import numpy as np
from typing import Any
from PIL import Image
import yaml

# helpers
from .collision_models import get_vertices

# zooming constants
ZOOM_IN_FACTOR = 1.2
ZOOM_OUT_FACTOR = 1 / ZOOM_IN_FACTOR

# vehicle shape constants
CAR_LENGTH = 0.58
CAR_WIDTH = 0.31


class CarShape:
    """
    Custom shape class for rendering cars as quads using pyglet 2.x shapes API.
    """

    def __init__(self, vertices: list[float], color: tuple[int, int, int], batch: pyglet.graphics.Batch):
        """
        Create a car shape from 4 corner vertices.
        
        Args:
            vertices: List of 8 floats [x1, y1, x2, y2, x3, y3, x4, y4]
            color: RGB tuple (r, g, b)
            batch: pyglet Batch to add the shape to
        """
        self._batch = batch
        self._color = color
        self._vertices_data = vertices
        
        # Create two triangles to form a quad (pyglet 2.x doesn't have GL_QUADS directly)
        # Triangle 1: vertices 0, 1, 2
        # Triangle 2: vertices 0, 2, 3
        self._triangles: list[pyglet.shapes.Triangle] = []
        self._update_triangles()

    def _update_triangles(self):
        """Update the triangle shapes based on current vertices."""
        # Delete old triangles
        for tri in self._triangles:
            tri.delete()
        self._triangles = []
        
        v = self._vertices_data
        if len(v) >= 8:
            # Triangle 1: v0, v1, v2
            tri1 = pyglet.shapes.Triangle(
                v[0], v[1],  # vertex 0
                v[2], v[3],  # vertex 1
                v[4], v[5],  # vertex 2
                color=self._color,
                batch=self._batch
            )
            # Triangle 2: v0, v2, v3
            tri2 = pyglet.shapes.Triangle(
                v[0], v[1],  # vertex 0
                v[4], v[5],  # vertex 2
                v[6], v[7],  # vertex 3
                color=self._color,
                batch=self._batch
            )
            self._triangles = [tri1, tri2]

    @property
    def vertices(self) -> list[float]:
        return self._vertices_data

    @vertices.setter
    def vertices(self, value: list[float]):
        self._vertices_data = value
        self._update_triangles()

    def delete(self):
        for tri in self._triangles:
            tri.delete()
        self._triangles = []


class EnvRenderer(pyglet.window.Window):
    """
    A window class inherited from pyglet.window.Window, handles the camera/projection interaction, resizing window, and rendering the environment
    """

    def __init__(self, width: int, height: int, *args: Any, **kwargs: Any):
        """
        Class constructor

        Args:
            width (int): width of the window
            height (int): height of the window

        Returns:
            None
        """
        conf = Config(sample_buffers=1, samples=4, depth_size=16, double_buffer=True)
        super().__init__(
            width, height, config=conf, resizable=True, vsync=False, *args, **kwargs
        )

        # Set background color
        pyglet.gl.glClearColor(9 / 255, 32 / 255, 87 / 255, 1.0)

        # initialize camera values
        self.left = -width / 2
        self.right = width / 2
        self.bottom = -height / 2
        self.top = height / 2
        self.zoom_level = 1.2
        self.zoomed_width = width
        self.zoomed_height = height

        # Camera offset for panning
        self.camera_x = 0.0
        self.camera_y = 0.0

        # current batch that keeps track of all graphics
        self.batch = pyglet.graphics.Batch()

        # current env map
        self.map_points = None
        self.map_point_shapes: list[pyglet.shapes.Circle] = []

        # current env agent poses, (num_agents, 3), columns are (x, y, theta)
        self.poses = None

        # current env agent vertices, (num_agents, 4, 2), 2nd and 3rd dimensions are the 4 corners in 2D
        self.vertices = None

        # car shapes
        self.cars: list[CarShape] = []

        # current score label
        self.score_label = pyglet.text.Label(
            "Lap Time: {laptime:.2f}, Ego Lap Count: {count:.0f}".format(
                laptime=0.0, count=0.0
            ),
            font_size=36,
            x=width // 2,
            y=50,
            anchor_x="center",
            anchor_y="center",
            color=(255, 255, 255, 255),
        )

        self.fps_display = pyglet.window.FPSDisplay(self)

    def update_map(self, map_path: str, map_ext: str) -> None:
        """
        Update the map being drawn by the renderer. Converts image to a list of 3D points representing each obstacle pixel in the map.

        Args:
            map_path (str): absolute path to the map without extensions
            map_ext (str): extension for the map image file

        Returns:
            None
        """

        # load map metadata
        with open(map_path + ".yaml", "r") as yaml_stream:
            try:
                map_metadata = yaml.safe_load(yaml_stream)
                map_resolution = map_metadata["resolution"]
                origin = map_metadata["origin"]
                origin_x = origin[0]
                origin_y = origin[1]
            except yaml.YAMLError as ex:
                print(ex)

        # load map image
        map_img = np.array(
            Image.open(map_path + map_ext).transpose(Image.FLIP_TOP_BOTTOM)
        ).astype(np.float64)
        map_height = map_img.shape[0]
        map_width = map_img.shape[1]

        # convert map pixels to coordinates
        range_x = np.arange(map_width)
        range_y = np.arange(map_height)
        map_x, map_y = np.meshgrid(range_x, range_y)
        map_x = (map_x * map_resolution + origin_x).flatten()
        map_y = (map_y * map_resolution + origin_y).flatten()
        map_z = np.zeros(map_y.shape)
        map_coords = np.vstack((map_x, map_y, map_z))

        # mask and only leave the obstacle points
        map_mask = map_img == 0.0
        map_mask_flat = map_mask.flatten()
        map_points = 50.0 * map_coords[:, map_mask_flat].T
        
        # Clear old map point shapes
        for shape in self.map_point_shapes:
            shape.delete()
        self.map_point_shapes = []
        
        # Create circle shapes for map points (pyglet 2.x compatible)
        # Using small circles to represent points
        point_color = (183, 193, 222)
        for i in range(map_points.shape[0]):
            point = pyglet.shapes.Circle(
                x=map_points[i, 0],
                y=map_points[i, 1],
                radius=1.0,  # Small radius for point-like appearance
                color=point_color,
                batch=self.batch
            )
            self.map_point_shapes.append(point)
        
        self.map_points = map_points

    def on_resize(self, width: int, height: int) -> None:
        """
        Callback function on window resize, overrides inherited method, and updates camera values on top of the inherited on_resize() method.

        Potential improvements on current behavior: zoom/pan resets on window resize.

        Args:
            width (int): new width of window
            height (int): new height of window

        Returns:
            None
        """

        # call overrided function
        super().on_resize(width, height)

        # update camera value
        (width, height) = self.get_size()
        self.left = -self.zoom_level * width / 2
        self.right = self.zoom_level * width / 2
        self.bottom = -self.zoom_level * height / 2
        self.top = self.zoom_level * height / 2
        self.zoomed_width = self.zoom_level * width
        self.zoomed_height = self.zoom_level * height
        
        # Update score label position
        self.score_label.x = width // 2
        self.score_label.y = 50

    def on_mouse_drag(
        self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int
    ) -> None:
        """
        Callback function on mouse drag, overrides inherited method.

        Args:
            x (int): Distance in pixels from the left edge of the window.
            y (int): Distance in pixels from the bottom edge of the window.
            dx (int): Relative X position from the previous mouse position.
            dy (int): Relative Y position from the previous mouse position.
            buttons (int): Bitwise combination of the mouse buttons currently pressed.
            modifiers (int): Bitwise combination of any keyboard modifiers currently active.

        Returns:
            None
        """

        # pan camera
        self.camera_x += dx * self.zoom_level
        self.camera_y += dy * self.zoom_level
        self.left -= dx * self.zoom_level
        self.right -= dx * self.zoom_level
        self.bottom -= dy * self.zoom_level
        self.top -= dy * self.zoom_level

    def on_mouse_scroll(self, x: int, y: int, dx: float, dy: float) -> None:
        """
        Callback function on mouse scroll, overrides inherited method.

        Args:
            x (int): Distance in pixels from the left edge of the window.
            y (int): Distance in pixels from the bottom edge of the window.
            dx (float): Amount of movement on the horizontal axis.
            dy (float): Amount of movement on the vertical axis.

        Returns:
            None
        """

        # Get scale factor
        f = ZOOM_IN_FACTOR if dy > 0 else ZOOM_OUT_FACTOR if dy < 0 else 1

        # If zoom_level is in the proper range
        if 0.01 < self.zoom_level * f < 10:
            self.zoom_level *= f

            (width, height) = self.get_size()

            mouse_x = x / width
            mouse_y = y / height

            mouse_x_in_world = self.left + mouse_x * self.zoomed_width
            mouse_y_in_world = self.bottom + mouse_y * self.zoomed_height

            self.zoomed_width *= f
            self.zoomed_height *= f

            self.left = mouse_x_in_world - mouse_x * self.zoomed_width
            self.right = mouse_x_in_world + (1 - mouse_x) * self.zoomed_width
            self.bottom = mouse_y_in_world - mouse_y * self.zoomed_height
            self.top = mouse_y_in_world + (1 - mouse_y) * self.zoomed_height

    def on_close(self) -> None:
        """
        Callback function when the 'x' is clicked on the window, overrides inherited method. Also throws exception to end the python program when in a loop.

        Args:
            None

        Returns:
            None

        Raises:
            Exception: with a message that indicates the rendering window was closed
        """

        super().on_close()
        raise Exception("Rendering window was closed.")

    def on_draw(self) -> None:
        """
        Function when the pyglet is drawing. The function draws the batch created that includes the map points, the agent polygons, and the information text, and the fps display.

        Args:
            None

        Returns:
            None
        """

        # if map and poses doesn't exist, raise exception
        if self.map_points is None:
            raise Exception("Map not set for renderer.")
        if self.poses is None:
            raise Exception("Agent poses not updated for renderer.")

        # Clear window
        self.clear()

        # Set up the projection for 2D rendering with camera
        self.projection = pyglet.math.Mat4.orthogonal_projection(
            self.left, self.right, self.bottom, self.top, -1, 1
        )

        # Draw all batches
        with self.window_block():
            self.batch.draw()
        
        # Draw UI elements (score label and fps) in screen space
        self.score_label.draw()
        self.fps_display.draw()

    def window_block(self):
        """Context manager for setting window projection"""
        return _ProjectionContext(self)

    def update_obs(self, obs: dict[str, Any]) -> None:
        """
        Updates the renderer with the latest observation from the gym environment, including the agent poses, and the information text.

        Args:
            obs (dict): observation dict from the gym env

        Returns:
            None
        """

        self.ego_idx = obs["ego_idx"]
        poses_x = obs["poses_x"]
        poses_y = obs["poses_y"]
        poses_theta = obs["poses_theta"]

        num_agents = len(poses_x)
        if self.poses is None:
            self.cars = []
            for i in range(num_agents):
                if i == self.ego_idx:
                    vertices_np = get_vertices(
                        np.array([0.0, 0.0, 0.0]), CAR_LENGTH, CAR_WIDTH
                    )
                    vertices = list(vertices_np.flatten())
                    car = CarShape(
                        vertices=vertices,
                        color=(172, 97, 185),  # Ego car color
                        batch=self.batch
                    )
                    self.cars.append(car)
                else:
                    vertices_np = get_vertices(
                        np.array([0.0, 0.0, 0.0]), CAR_LENGTH, CAR_WIDTH
                    )
                    vertices = list(vertices_np.flatten())
                    car = CarShape(
                        vertices=vertices,
                        color=(99, 52, 94),  # Other car color
                        batch=self.batch
                    )
                    self.cars.append(car)

        poses = np.stack((poses_x, poses_y, poses_theta)).T
        for j in range(poses.shape[0]):
            vertices_np = 50.0 * get_vertices(poses[j, :], CAR_LENGTH, CAR_WIDTH)
            vertices = list(vertices_np.flatten())
            self.cars[j].vertices = vertices
        self.poses = poses

        self.score_label.text = (
            "Lap Time: {laptime:.2f}, Ego Lap Count: {count:.0f}".format(
                laptime=obs["lap_times"][0], count=obs["lap_counts"][obs["ego_idx"]]
            )
        )


class _ProjectionContext:
    """Context manager for temporarily setting window projection."""
    
    def __init__(self, window: EnvRenderer):
        self.window = window
    
    def __enter__(self):
        # Store old view and set new projection
        self.window.view = pyglet.math.Mat4()
        return self
    
    def __exit__(self, *args):
        pass
