import cv2
import numpy as np

line_points = []
goal_side = None  # 'left' or 'right'


def mouse_callback(event, x, y, flags, param):
    global line_points
    if event == cv2.EVENT_LBUTTONDOWN and len(line_points) < 2:
        line_points.append((x, y))


def reset_line():
    global line_points, goal_side
    line_points = []
    goal_side = None


def is_line_complete():
    return len(line_points) == 2


def extend_line_to_edges(p1, p2, img_shape):
    """두 점으로 정의된 선을 이미지 경계까지 연장"""
    h, w = img_shape[:2]
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])

    if abs(x2 - x1) < 1e-6:  # 수직선
        return (int(x1), 0), (int(x1), h)

    slope = (y2 - y1) / (x2 - x1)
    y_left = int(slope * (0 - x1) + y1)
    y_right = int(slope * (w - x1) + y1)
    return (0, y_left), (w, y_right)


def point_side(p1, p2, point):
    """
    선(p1→p2)에 대해 point가 어느 쪽인지 반환
    양수: 왼쪽, 음수: 오른쪽 (수학적 기준)
    """
    return ((p2[0] - p1[0]) * (point[1] - p1[1]) -
            (p2[1] - p1[1]) * (point[0] - p1[0]))


def draw_line_on_image(image, p1, p2, color=(0, 220, 220), thickness=2):
    edge_p1, edge_p2 = extend_line_to_edges(p1, p2, image.shape)
    cv2.line(image, edge_p1, edge_p2, color, thickness)
    cv2.circle(image, p1, 6, (0, 255, 0), -1)
    cv2.circle(image, p2, 6, (0, 255, 0), -1)
    return image


def select_line_interactively(image):
    """
    사용자가 이미지에서 2번 클릭해 오프사이드 라인 지정
    Returns: (p1, p2, goal_side)
    """
    global line_points, goal_side
    reset_line()

    display = image.copy()
    window = "Offside Line Setup (클릭 2번으로 라인 지정)"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, mouse_callback)

    print("[1/2] 이미지 위에서 오프사이드 라인의 두 점을 클릭하세요.")

    while True:
        frame = display.copy()

        # 클릭한 점 표시
        for pt in line_points:
            cv2.circle(frame, pt, 6, (0, 255, 0), -1)

        # 두 점이 찍히면 라인 미리보기
        if len(line_points) == 2:
            draw_line_on_image(frame, line_points[0], line_points[1])
            cv2.putText(frame, "L: 골대 왼쪽 | R: 골대 오른쪽 | Enter: 확인",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            remaining = 2 - len(line_points)
            cv2.putText(frame, f"클릭 {remaining}번 남음",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow(window, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):
            reset_line()
            display = image.copy()

        if len(line_points) == 2 and goal_side is None:
            if key == ord('l'):
                goal_side = 'left'
            elif key == ord('r') and len(line_points) < 2:
                reset_line()
            elif key == ord('r') and len(line_points) == 2:
                goal_side = 'right'

        if len(line_points) == 2 and goal_side is not None:
            if key == 13:  # Enter
                break

    cv2.destroyWindow(window)
    return line_points[0], line_points[1], goal_side


def is_on_goal_side(p1, p2, point, goal_side):
    """
    point가 골대 방향(오프사이드 위치)에 있는지 반환.
    라인 클릭 순서와 무관하게 동작하도록 y보간으로 x비교.
    """
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    px, py = float(point[0]), float(point[1])

    if abs(y2 - y1) < 1e-6:
        # 수평선: x로 비교
        line_x = x1
    else:
        # point의 y 위치에서 라인의 x 보간
        t = (py - y1) / (y2 - y1)
        line_x = x1 + t * (x2 - x1)

    if goal_side == 'left':
        return px < line_x   # 라인보다 왼쪽 = 골대 방향 = 오프사이드
    else:
        return px > line_x   # 라인보다 오른쪽 = 골대 방향 = 오프사이드
