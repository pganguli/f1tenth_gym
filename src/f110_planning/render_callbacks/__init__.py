def camera_tracking_callback(env_renderer):
    """
    Update camera to follow car
    """
    e = env_renderer
    x = e.cars[0].vertices[::2]
    y = e.cars[0].vertices[1::2]
    top, bottom, left, right = max(y), min(y), min(x), max(x)
    e.left = left - 500
    e.right = right + 500
    e.top = top + 500
    e.bottom = bottom - 500
