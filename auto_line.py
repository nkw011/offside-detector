"""
Phase 2: 수비수 자동 감지 후 오프사이드 라인 제안
YOLOv8-pose + BLIP으로 수비 팀의 두 번째 선수 위치를 자동 계산

사용법:
    python auto_line.py --image <이미지경로> --defend <A|B> --goal <left|right>
"""

import argparse
import sys
import cv2
import numpy as np

from models.pose_detector import PoseDetector
from models.team_classifier import TeamClassifier
from models.reasoner import OffsideReasoner
from utils.line_tools import is_on_goal_side
from utils.visualizer import draw_result


def get_second_last_defender_x(detections, defending_team, goal_side):
    """
    수비 팀 선수들 중 골대와 두 번째로 가까운 선수의 x좌표 반환
    (골키퍼 = 첫 번째, 세컨드 디펜더 = 오프사이드 기준)
    """
    defenders = [d for d in detections if d.get('team') == defending_team]
    if len(defenders) < 2:
        return None, defenders

    # 발목 x좌표 기준 정렬
    def foot_x(d):
        return d['forward_foot'][0]

    if goal_side == 'left':
        # 골대가 왼쪽 → x가 작을수록 골대에 가까움
        defenders_sorted = sorted(defenders, key=foot_x)
    else:
        # 골대가 오른쪽 → x가 클수록 골대에 가까움
        defenders_sorted = sorted(defenders, key=foot_x, reverse=True)

    # 두 번째 수비수
    second_defender = defenders_sorted[1] if len(defenders_sorted) >= 2 else defenders_sorted[0]
    return second_defender['forward_foot'][0], defenders_sorted


def propose_line(image, second_x):
    """
    second_x를 기반으로 수직에 가까운 라인 제안 (이미지 높이 방향)
    반환: (p1, p2)
    """
    h = image.shape[0]
    p1 = (second_x, 0)
    p2 = (second_x, h)
    return p1, p2


def judge_offside(detections, p1, p2, goal_side, team_attacking):
    for det in detections:
        if det.get('team') != team_attacking:
            det['is_offside'] = False
            continue
        foot = det['forward_foot']
        det['is_offside'] = is_on_goal_side(p1, p2, foot, goal_side)
    return detections


def parse_args():
    parser = argparse.ArgumentParser(description="자동 라인 제안 오프사이드 판독기")
    parser.add_argument("--image", required=True)
    parser.add_argument("--defend", default="B", choices=["A", "B"], help="수비 팀")
    parser.add_argument("--goal", default="left", choices=["left", "right"], help="골대 방향")
    parser.add_argument("--skip-qwen", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    attacking_team = "A" if args.defend == "B" else "B"

    image = cv2.imread(args.image)
    if image is None:
        print(f"[Error] 이미지 없음: {args.image}")
        sys.exit(1)

    print(f"[1/5] 이미지 로드: {image.shape[1]}x{image.shape[0]}")

    print("\n[모델 로딩]")
    pose_detector = PoseDetector(model_size="n")
    team_classifier = TeamClassifier()
    reasoner = None if args.skip_qwen else OffsideReasoner()

    print("\n[2/5] 선수 탐지 + Pose Estimation")
    detections = pose_detector.detect(image)
    print(f"  탐지: {len(detections)}명")

    print("\n[3/5] BLIP 팀 분류")
    detections, team_map = team_classifier.classify_teams(image, detections)
    print(f"  팀 매핑: {team_map}")

    print(f"\n[4/5] 수비수 자동 감지 → 라인 제안 (수비 팀: {args.defend})")
    second_x, defenders_sorted = get_second_last_defender_x(
        detections, args.defend, args.goal
    )

    if second_x is None:
        print("[Warning] 수비수가 2명 이상 탐지되지 않았습니다. 첫 번째 수비수 사용.")
        if not defenders_sorted:
            print("[Error] 수비수 없음. 종료.")
            sys.exit(1)
        second_x = defenders_sorted[0]['forward_foot'][0]

    print(f"  두 번째 수비수 x좌표: {second_x}px → 라인 자동 설정")
    p1, p2 = propose_line(image, second_x)

    # 사용자 확인 / 미세 조정
    preview = image.copy()
    cv2.line(preview, p1, p2, (0, 220, 220), 2)
    cv2.putText(preview, f"제안 라인 x={second_x} | Enter 확정 | 방향키로 조정",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.imshow("라인 제안 (Enter: 확정, ←/→: 조정)", preview)

    offset = 0
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 13:  # Enter
            break
        elif key == 81 or key == ord('a'):  # 왼쪽 화살표
            offset -= 5
        elif key == 83 or key == ord('d'):  # 오른쪽 화살표
            offset += 5

        adjusted_x = second_x + offset
        p1, p2 = propose_line(image, adjusted_x)
        preview = image.copy()
        cv2.line(preview, p1, p2, (0, 220, 220), 2)
        cv2.putText(preview, f"x={adjusted_x} (offset={offset:+d}) | Enter 확정",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("라인 제안 (Enter: 확정, ←/→: 조정)", preview)

    cv2.destroyAllWindows()

    print(f"\n[5/5] 오프사이드 판독 (공격 팀: {attacking_team})")
    detections = judge_offside(detections, p1, p2, args.goal, attacking_team)
    offside_count = sum(1 for d in detections if d.get('is_offside'))
    print(f"  오프사이드: {offside_count}명")

    annotated = draw_result(image, p1, p2, detections)

    if reasoner:
        verdict = reasoner.analyze(annotated, detections, args.goal, attacking_team)
        print("\n판독 근거:\n", verdict)

    cv2.imshow("Auto Offside Detector", annotated)
    out_path = args.image.replace(".", "_auto_result.")
    cv2.imwrite(out_path, annotated)
    print(f"결과 저장: {out_path}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
