"""
반자동 오프사이드 판독기
Foundation Models: YOLOv8-pose / BLIP VQA / Qwen2.5-VL

사용법:
    python main.py --image <이미지경로> --team <A|B>

단계:
    1. 이미지 로드
    2. 마우스 클릭 2번으로 오프사이드 라인 지정
    3. YOLOv8-pose로 선수 탐지 + keypoint 추출
    4. BLIP으로 팀 분류
    5. 기하학적 판독 (발목 vs 라인)
    6. Qwen2.5-VL로 판독 근거 생성
"""

import argparse
import sys
import cv2
import numpy as np

from models.pose_detector import PoseDetector
from models.team_classifier import TeamClassifier
from models.reasoner import OffsideReasoner
from utils.line_tools import select_line_interactively, is_on_goal_side
from utils.visualizer import draw_result


def parse_args():
    parser = argparse.ArgumentParser(description="반자동 오프사이드 판독기")
    parser.add_argument("--image", required=True, help="입력 이미지 경로")
    parser.add_argument("--team", default="A", choices=["A", "B"],
                        help="공격 팀 (A 또는 B)")
    parser.add_argument("--skip-qwen", action="store_true",
                        help="Qwen 모델 로딩 스킵 (빠른 데모용)")
    parser.add_argument("--skip-blip", action="store_true",
                        help="BLIP 스킵 후 수동 팀 지정")
    return parser.parse_args()


def manual_team_assignment(detections, image):
    """BLIP 스킵 시: 사용자가 직접 클릭해서 팀 지정"""
    print("\n[수동 팀 지정] 각 선수 bbox를 보고 팀을 A/B로 입력하세요.")
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det['bbox']
        team = input(f"  선수 {i+1} (bbox: {det['bbox']}): ").strip().upper()
        det['team'] = team if team in ('A', 'B') else 'unknown'
    return detections


def judge_offside(detections, p1, p2, goal_side, team_attacking):
    """기하학적 판독: 공격 팀의 각 선수가 라인 앞에 있는지 확인"""
    for det in detections:
        if det.get('team') != team_attacking:
            det['is_offside'] = False
            continue

        foot = det['forward_foot']
        if foot[0] == 0 and foot[1] == 0:
            det['is_offside'] = False
            continue

        det['is_offside'] = is_on_goal_side(p1, p2, foot, goal_side)

    return detections


def main():
    args = parse_args()

    # ── 이미지 로드 ──────────────────────────────────────────
    image = cv2.imread(args.image)
    if image is None:
        print(f"[Error] 이미지를 열 수 없습니다: {args.image}")
        sys.exit(1)
    print(f"[1/6] 이미지 로드: {image.shape[1]}x{image.shape[0]}")

    # ── 모델 로딩 ────────────────────────────────────────────
    print("\n[모델 로딩]")
    pose_detector = PoseDetector(model_size="n")
    team_classifier = None if args.skip_blip else TeamClassifier()
    reasoner = None if args.skip_qwen else OffsideReasoner()

    # ── 라인 지정 ────────────────────────────────────────────
    print("\n[2/6] 오프사이드 라인 지정")
    print("  - 클릭 2번으로 라인 양 끝점 지정")
    print("  - 이후 L (골대 왼쪽) 또는 R (골대 오른쪽) 키 입력")
    print("  - Enter로 확정, R키로 리셋")
    p1, p2, goal_side = select_line_interactively(image)
    print(f"  라인: {p1} → {p2}, 골대 방향: {goal_side}")

    # ── 선수 탐지 + Pose Estimation ─────────────────────────
    print("\n[3/6] YOLOv8-pose: 선수 탐지 + 키포인트 추출")
    detections = pose_detector.detect(image)
    print(f"  탐지된 선수: {len(detections)}명")

    if not detections:
        print("[Warning] 선수가 탐지되지 않았습니다.")
        cv2.imshow("Result", image)
        cv2.waitKey(0)
        return

    # ── 팀 분류 ─────────────────────────────────────────────
    print("\n[4/6] 팀 분류")
    if team_classifier:
        print("  BLIP VQA로 유니폼 색상 분석 중...")
        detections, team_map = team_classifier.classify_teams(image, detections)
        print(f"  팀 매핑: {team_map}")
    else:
        detections = manual_team_assignment(detections, image)

    # ── 오프사이드 판독 ──────────────────────────────────────
    print(f"\n[5/6] 오프사이드 판독 (공격 팀: {args.team})")
    detections = judge_offside(detections, p1, p2, goal_side, args.team)
    offside_count = sum(1 for d in detections if d.get('is_offside'))
    print(f"  결과: 오프사이드 {offside_count}명")

    # ── Qwen 판독 근거 ───────────────────────────────────────
    print("\n[6/6] Qwen2.5-VL 판독 근거 생성")
    annotated = draw_result(image, p1, p2, detections)
    verdict = ""
    if reasoner:
        verdict = reasoner.analyze(annotated, detections, goal_side, args.team)
    else:
        verdict_lines = []
        for d in detections:
            if d.get('team') == args.team:
                status = "OFFSIDE" if d.get('is_offside') else "onside"
                verdict_lines.append(f"  발위치 {d['forward_foot']}: {status}")
        verdict = "\n".join(verdict_lines)

    print("\n" + "="*50)
    print("판독 결과")
    print("="*50)
    print(verdict)
    print("="*50)

    # ── 결과 출력 ────────────────────────────────────────────
    cv2.imshow("Offside Detector", annotated)
    out_path = args.image.replace(".", "_result.")
    cv2.imwrite(out_path, annotated)
    print(f"\n결과 이미지 저장: {out_path}")
    print("아무 키나 누르면 종료")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
