import gradio as gr
import cv2
import numpy as np
from PIL import Image
import traceback

from models.pose_detector import PoseDetector
from models.team_classifier import TeamClassifier
from models.reasoner import OffsideReasoner
from utils.line_tools import extend_line_to_edges, is_on_goal_side
from utils.visualizer import draw_detections, make_all_crops

print("모델 로딩 중...")
pose_detector   = PoseDetector(model_size="s")
team_classifier = TeamClassifier()
reasoner        = OffsideReasoner()
print("모델 로딩 완료")

# ── 전역 상태 ──────────────────────────────────────────────
g_clean_np     = None   # 원본 이미지 BGR
g_attack_marks = []     # [(x,y)]  공격팀 seed (1개로 충분)
g_defend_marks = []     # [(x,y)]  수비팀 seed (1개로 충분)


# ── 렌더러 ─────────────────────────────────────────────────
def _render(line_pts=None, detections=None):
    if g_clean_np is None:
        return None
    canvas = cv2.cvtColor(g_clean_np.copy(), cv2.COLOR_BGR2RGB)

    # 자동 계산된 오프사이드 라인 (판독 후)
    if line_pts and len(line_pts) == 2:
        ep1, ep2 = extend_line_to_edges(line_pts[0], line_pts[1], canvas.shape)
        cv2.line(canvas, ep1, ep2, (0, 220, 220), 2)

    # 공격팀 seed 마커 (주황)
    for pt in g_attack_marks:
        cv2.circle(canvas, pt, 14, (255, 140, 0), -1)
        cv2.circle(canvas, pt, 16, (255, 255, 255), 2)
        cv2.putText(canvas, "A", (pt[0]-6, pt[1]+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # 수비팀 seed 마커 (파랑)
    for pt in g_defend_marks:
        cv2.circle(canvas, pt, 14, (30, 100, 255), -1)
        cv2.circle(canvas, pt, 16, (255, 255, 255), 2)
        cv2.putText(canvas, "B", (pt[0]-6, pt[1]+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    if detections:
        draw_detections(canvas, detections)

    return Image.fromarray(canvas)


# ── 업로드 ─────────────────────────────────────────────────
def on_upload(img_pil):
    global g_clean_np, g_attack_marks, g_defend_marks
    if img_pil is None:
        return None, "이미지를 업로드하세요."
    g_clean_np     = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    g_attack_marks = []
    g_defend_marks = []
    return _render(), "✅ 업로드 완료.\n공격팀(A) 선수 1명, 수비팀(B) 선수 1명을 클릭하세요."


# ── 클릭 핸들러 ────────────────────────────────────────────
def on_click(team_choice, evt: gr.SelectData):
    global g_attack_marks, g_defend_marks
    try:
        if g_clean_np is None:
            return None, "이미지를 먼저 업로드하세요."

        x, y = int(evt.index[0]), int(evt.index[1])

        if team_choice == "A — 공격팀":
            g_attack_marks = [(x, y)]  # seed는 1개면 충분
            msg = f"🔴 공격팀 A 선수 지정 ({x},{y})"
        else:
            g_defend_marks = [(x, y)]
            msg = f"🔵 수비팀 B 선수 지정 ({x},{y})"

        ready = len(g_attack_marks) > 0 and len(g_defend_marks) > 0
        if ready:
            msg += "\n✅ 준비 완료! '판독 시작'을 누르세요."
        return _render(), msg

    except Exception as e:
        print(traceback.format_exc())
        return _render(), f"에러: {e}"


# ── 마커 취소 ──────────────────────────────────────────────
def undo_last():
    global g_attack_marks, g_defend_marks
    if g_defend_marks:
        g_defend_marks = []
        return _render(), "↩ 수비팀 B 마커 취소"
    elif g_attack_marks:
        g_attack_marks = []
        return _render(), "↩ 공격팀 A 마커 취소"
    return _render(), "취소할 항목 없음."


# ── 오프사이드 라인 자동 계산 ──────────────────────────────
def _auto_offside_line(dets, goal_side, img_shape):
    """
    수비팀 선수 중 두 번째로 골대에 가까운 선수 위치 → 수직 오프사이드 라인 반환
    (FIFA VAR 규정: 두 번째 마지막 수비수)
    """
    defenders = [d for d in dets if d['team'] == 'B']
    if not defenders:
        # 수비수 없으면 이미지 중앙
        mid_x = img_shape[1] // 2
        return (mid_x, 0), (mid_x, img_shape[0])

    # 골대 방향으로 가장 앞에 있는 순서 정렬
    if goal_side == 'left':
        defenders_sorted = sorted(defenders, key=lambda d: d['forward_foot'][0])
    else:
        defenders_sorted = sorted(defenders, key=lambda d: d['forward_foot'][0], reverse=True)

    # 골대 방향으로 가장 앞에 있는 수비수
    ref = defenders_sorted[0]
    lx = ref['forward_foot'][0]
    return (lx, 0), (lx, img_shape[0])


# ── 판독 ───────────────────────────────────────────────────
def run_offside(goal_side):
    try:
        if g_clean_np is None:
            return _render(), [], "이미지를 업로드하세요."
        if not g_attack_marks:
            return _render(), [], "공격팀(A) 선수를 클릭하세요."
        if not g_defend_marks:
            return _render(), [], "수비팀(B) 선수를 클릭하세요."

        # [Model 1] YOLOv8-pose — 선수 탐지 + keypoint
        dets = pose_detector.detect(g_clean_np)
        if not dets:
            return _render(), [], "선수를 탐지하지 못했습니다."
        for det in dets:
            det['is_offside'] = False

        # [Model 2] BLIP + k-means — 클릭 위치 픽셀 색상 기준으로 자동 팀 배정
        dets = team_classifier.auto_assign_teams(
            g_clean_np, dets,
            attack_xy=g_attack_marks[0],
            defend_xy=g_defend_marks[0],
        )
        a_count = sum(1 for d in dets if d['team'] == 'A')
        b_count = sum(1 for d in dets if d['team'] == 'B')

        # forward_point: 팔 제외 전신 중 골대 방향 최전방 부위 (FIFA 규정)
        OFFSIDE_KP = [0, 5, 6, 11, 12, 13, 14, 15, 16]
        for det in dets:
            kp = det['keypoints']
            valid = [(kp[i][0], kp[i][1]) for i in OFFSIDE_KP if kp[i][2] > 0.2]
            if not valid:
                x1b, y1b, x2b, y2b = det['bbox']
                det['forward_foot'] = (int((x1b+x2b)/2), int(y2b))
            else:
                best = (min if goal_side == 'left' else max)(valid, key=lambda p: p[0])
                det['forward_foot'] = (int(best[0]), int(best[1]))

        # 오프사이드 라인 자동 계산 (2번째 수비수 위치)
        p1, p2 = _auto_offside_line(dets, goal_side, g_clean_np.shape)

        # 오프사이드 판정 (공격팀 A만)
        for det in dets:
            if det['team'] == 'A':
                det['is_offside'] = is_on_goal_side(p1, p2, det['forward_foot'], goal_side)

        offside_n = sum(1 for d in dets if d.get('is_offside'))

        # 결과 이미지 렌더
        canvas = cv2.cvtColor(g_clean_np.copy(), cv2.COLOR_BGR2RGB)
        ep1, ep2 = extend_line_to_edges(p1, p2, canvas.shape)
        cv2.line(canvas, ep1, ep2, (0, 220, 220), 2)
        draw_detections(canvas, dets)
        verdict = f"OFFSIDE x{offside_n}" if offside_n else "ONSIDE"
        v_color = (255, 0, 0) if offside_n else (0, 255, 0)
        cv2.putText(canvas, verdict, (10, canvas.shape[0]-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, v_color, 3)
        result_pil = Image.fromarray(canvas)

        # 공격팀 A 크롭
        img_rgb = cv2.cvtColor(g_clean_np, cv2.COLOR_BGR2RGB)
        crops = make_all_crops(img_rgb, dets, p1, p2, 'A')

        # [Model 3] Qwen3-VL-8B — 판독 근거
        result_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        qwen_text = reasoner.analyze(result_bgr, dets, goal_side, 'A')

        return result_pil, crops, qwen_text

    except Exception as e:
        print(traceback.format_exc())
        return _render(), [], f"판독 에러: {e}"


def reset_all():
    global g_clean_np, g_attack_marks, g_defend_marks
    g_clean_np     = None
    g_attack_marks = []
    g_defend_marks = []
    return None, "초기화 완료."


# ── UI ─────────────────────────────────────────────────────
with gr.Blocks(title="반자동 오프사이드 판독기") as demo:
    gr.Markdown(
        "# ⚽ 반자동 오프사이드 판독기\n"
        "`YOLOv8s-pose` · `BLIP VQA` · `Qwen3-VL-8B`\n\n"
        "**사용법:** ① 이미지 업로드 → ② 공격팀(A) 선수 클릭 → ③ 수비팀(B) 선수 클릭 → ④ 판독 시작"
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="이미지 (선수 클릭으로 팀 지정)",
                type="pil", interactive=True,
            )
            team_choice = gr.Radio(
                ["A — 공격팀", "B — 수비팀"],
                value="A — 공격팀",
                label="클릭할 팀 선택",
            )
            status_box = gr.Textbox(
                label="상태",
                value="이미지를 업로드하세요.",
                interactive=False, lines=3,
            )
            with gr.Row():
                undo_btn  = gr.Button("↩ 마지막 취소", size="sm")
                reset_btn = gr.Button("전체 초기화", variant="stop", size="sm")

            gr.Markdown("---")
            goal_side = gr.Radio(["left", "right"], label="골대 방향", value="left")
            run_btn   = gr.Button("판독 시작", variant="primary", size="lg")

        with gr.Column(scale=1):
            output_image  = gr.Image(label="판독 결과", type="pil")
            crops_gallery = gr.Gallery(
                label="공격팀(A) 선수 확대  (Pose + 오프사이드 여부)",
                columns=3, height=350,
            )
            output_log = gr.Textbox(
                label="분석 로그 / Qwen 판독 근거",
                lines=12, interactive=False,
            )

    # 이벤트
    input_image.upload(
        fn=on_upload,
        inputs=[input_image],
        outputs=[input_image, status_box],
    )
    input_image.select(
        fn=on_click,
        inputs=[team_choice],
        outputs=[input_image, status_box],
    )
    undo_btn.click(
        fn=undo_last,
        outputs=[input_image, status_box],
    )
    run_btn.click(
        fn=run_offside,
        inputs=[goal_side],
        outputs=[output_image, crops_gallery, output_log],
    )
    reset_btn.click(
        fn=reset_all,
        outputs=[input_image, status_box],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False)
