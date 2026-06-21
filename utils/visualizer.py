import cv2
import numpy as np
from PIL import Image
from .line_tools import draw_line_on_image, extend_line_to_edges

TEAM_COLORS = {
    'A': (0, 200, 255),
    'B': (255, 180, 0),
    'unknown': (180, 180, 180),
}
OFFSIDE_COLOR = (0, 0, 255)
ONSIDE_COLOR  = (0, 255, 0)
SELECTED_COLOR = (255, 0, 255)  # 선택된 선수 (마젠타)


def draw_skeleton(image, keypoints, color=(255, 255, 0), offset=(0, 0)):
    CONNECTIONS = [
        (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 6), (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16),
    ]
    ox, oy = offset
    for a, b in CONNECTIONS:
        pa, pb = keypoints[a], keypoints[b]
        if pa[2] > 0.2 and pb[2] > 0.2:
            cv2.line(image,
                     (int(pa[0]) + ox, int(pa[1]) + oy),
                     (int(pb[0]) + ox, int(pb[1]) + oy),
                     color, 1)
    for kp in keypoints:
        if kp[2] > 0.2:
            cv2.circle(image, (int(kp[0]) + ox, int(kp[1]) + oy), 3, color, -1)


def draw_detections(image, detections, selected_idx=None):
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det['bbox']
        team      = det.get('team', 'unknown')
        is_offside  = det.get('is_offside', False)
        is_selected = (i == selected_idx)

        if is_selected:
            border_color = SELECTED_COLOR
        elif is_offside:
            border_color = OFFSIDE_COLOR
        else:
            border_color = TEAM_COLORS.get(team, (200, 200, 200))

        thickness = 4 if is_selected else 2
        cv2.rectangle(image, (x1, y1), (x2, y2), border_color, thickness)

        # 발목 점
        fp = det.get('forward_foot')
        if fp and fp[0] > 0:
            cv2.circle(image, (int(fp[0]), int(fp[1])), 7,
                       OFFSIDE_COLOR if is_offside else ONSIDE_COLOR, -1)

        # 스켈레톤
        kp = det.get('keypoints')
        if kp is not None:
            draw_skeleton(image, kp, color=border_color)

        # 라벨 배경 + 텍스트 (번호, 팀, 오프사이드 여부)
        jersey = det.get('jersey_color', '')[:6]
        if is_selected:
            label = f"[{i}] T{team} ★SCORER"
        elif is_offside:
            label = f"[{i}] T{team} OFFSIDE"
        else:
            label = f"[{i}] T{team} {jersey}"

        font_scale, font_thick = 0.75, 2
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
        lx, ly = x1, max(y1 - 6, th + 6)
        cv2.rectangle(image, (lx, ly - th - 6), (lx + tw + 6, ly + 2),
                      (0, 0, 0), -1)
        cv2.putText(image, label, (lx + 3, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, border_color, font_thick)

    return image


def draw_result(image, p1, p2, detections, selected_idx=None):
    out = image.copy()
    draw_line_on_image(out, p1, p2)
    draw_detections(out, detections, selected_idx=selected_idx)

    offside_players = [d for d in detections if d.get('is_offside')]
    result_str = f"OFFSIDE x{len(offside_players)}" if offside_players else "ONSIDE"
    color = OFFSIDE_COLOR if offside_players else ONSIDE_COLOR
    cv2.putText(out, result_str, (10, out.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    return out


def make_player_crop(image_np, det, p1, p2, pad=50):
    """
    오프사이드 라인 근처 선수 확대 크롭
    - 스켈레톤 + 발 위치 + 라인 + OFFSIDE/ONSIDE 판정 표시
    """
    x1, y1, x2, y2 = det['bbox']
    h, w = image_np.shape[:2]

    cx1 = max(0, x1 - pad)
    cy1 = max(0, y1 - pad)
    cx2 = min(w, x2 + pad)
    cy2 = min(h, y2 + pad)

    crop = image_np[cy1:cy2, cx1:cx2].copy()
    ch, cw = crop.shape[:2]

    # 오프사이드 라인을 크롭 좌표계로 변환
    lp1 = (p1[0] - cx1, p1[1] - cy1)
    lp2 = (p2[0] - cx1, p2[1] - cy1)
    ep1, ep2 = extend_line_to_edges(lp1, lp2, crop.shape)
    cv2.line(crop, ep1, ep2, (0, 220, 220), 2)

    # 스켈레톤 (크롭 좌표 오프셋 적용)
    kp = det.get('keypoints')
    is_offside = det.get('is_offside', False)
    skel_color = OFFSIDE_COLOR if is_offside else ONSIDE_COLOR

    if kp is not None:
        kp_shifted = kp.copy()
        kp_shifted[:, 0] -= cx1
        kp_shifted[:, 1] -= cy1
        draw_skeleton(crop, kp_shifted, color=skel_color)

    # 발 위치 마커
    fp = det.get('forward_foot')
    if fp and fp[0] > 0:
        fp_crop = (fp[0] - cx1, fp[1] - cy1)
        if 0 <= fp_crop[0] < cw and 0 <= fp_crop[1] < ch:
            cv2.circle(crop, fp_crop, 10, skel_color, -1)
            cv2.circle(crop, fp_crop, 12, (255, 255, 255), 2)

    # 판정 텍스트
    verdict = "OFFSIDE" if is_offside else "ONSIDE"
    color = OFFSIDE_COLOR if is_offside else ONSIDE_COLOR
    cv2.rectangle(crop, (0, 0), (cw, 32), (0, 0, 0), -1)
    cv2.putText(crop, verdict, (5, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    team = det.get('team', '?')
    jersey = det.get('jersey_color', '')
    sub = f"Team {team}  {jersey}"
    cv2.putText(crop, sub, (5, ch - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)

    return Image.fromarray(crop[..., ::-1])  # BGR→RGB


def make_all_crops(image_np, detections, p1, p2, attacking_team):
    """공격 팀 선수 전원 크롭 생성"""
    crops = []
    for det in detections:
        if det.get('team') != attacking_team:
            continue
        crop_img = make_player_crop(image_np, det, p1, p2)
        verdict = "OFFSIDE" if det.get('is_offside') else "ONSIDE"
        jersey = det.get('jersey_color', '')
        caption = f"{verdict} | {jersey}"
        crops.append((crop_img, caption))
    return crops
