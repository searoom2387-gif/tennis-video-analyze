COURT_WIDTH_M = 8.23
COURT_LENGTH_M = 23.77
SERVICE_LINE_FROM_NET_M = 6.40

Point = tuple[float, float]


def court_lines() -> list[tuple[Point, Point]]:
    """Singles court line segments in court coordinates (x: width, y: length, meters)."""
    w, length = COURT_WIDTH_M, COURT_LENGTH_M
    net_y = length / 2
    near_service_y = net_y - SERVICE_LINE_FROM_NET_M
    far_service_y = net_y + SERVICE_LINE_FROM_NET_M
    return [
        ((0.0, 0.0), (w, 0.0)),  # baseline
        ((0.0, length), (w, length)),  # baseline
        ((0.0, 0.0), (0.0, length)),  # singles sideline
        ((w, 0.0), (w, length)),  # singles sideline
        ((0.0, near_service_y), (w, near_service_y)),  # service line
        ((0.0, far_service_y), (w, far_service_y)),  # service line
        ((w / 2, near_service_y), (w / 2, far_service_y)),  # center service line
    ]
