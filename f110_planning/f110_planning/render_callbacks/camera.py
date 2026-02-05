from f110_gym.envs.rendering import EnvRenderer

def camera_tracking(env_renderer: EnvRenderer) -> None:
    """
    Update camera to follow car (ego car at index 0).
    """
    if not env_renderer.cars:
        return
        
    # Get ego car vertices
    # vertices are [x1, y1, x2, y2, x3, y3, x4, y4]
    v = env_renderer.cars[0].vertices
    x = v[::2]
    y = v[1::2]
    
    top, bottom, left, right = max(y), min(y), min(x), max(x)
    
    # Update camera boundaries with padding
    padding = 500
    env_renderer.left = left - padding
    env_renderer.right = right + padding
    env_renderer.top = top + padding
    env_renderer.bottom = bottom - padding
