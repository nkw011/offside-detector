import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

try:
    from qwen_vl_utils import process_vision_info
    QWEN_UTILS = True
except ImportError:
    QWEN_UTILS = False


class OffsideReasoner:
    """
    Qwen3-VL-8B를 사용한 오프사이드 판독 근거 생성
    로딩 실패 시 텍스트 기반 폴백
    """

    def __init__(self, model_id="Qwen/Qwen3-VL-8B-Instruct"):
        self.available = False
        print(f"[Reasoner] {model_id} 로딩 시도...")
        try:
            self.processor = AutoProcessor.from_pretrained(
                model_id, trust_remote_code=True
            )
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                trust_remote_code=True,
            )
            self.model.eval()
            self.available = True
            print("[Reasoner] 완료")
        except Exception as e:
            print(f"[Reasoner] 로딩 실패: {e}")
            print("[Reasoner] 텍스트 기반 폴백 모드로 전환")

    def analyze(self, annotated_image_np, detections, goal_side, team_attacking):
        """
        annotated_image_np: 라인 + 선수 탐지가 그려진 이미지
        detections: 탐지된 선수 리스트 (team, is_offside 포함)
        goal_side: 'left' or 'right'
        team_attacking: 'A' or 'B'
        """
        offside_players = [d for d in detections if d.get('is_offside')]
        onside_players = [d for d in detections
                         if d.get('team') == team_attacking and not d.get('is_offside')]

        # 텍스트 기반 요약 (항상 생성)
        summary = self._build_summary(offside_players, onside_players, team_attacking, goal_side)

        if not self.available:
            return summary

        # Qwen2.5-VL 비주얼 분석
        try:
            pil_img = Image.fromarray(annotated_image_np[..., ::-1])
            prompt = (
                f"This is a soccer offside analysis image. "
                f"The cyan line is the offside line. "
                f"Team {team_attacking} is attacking towards the {goal_side} side. "
                f"Players marked in red are detected as offside. "
                f"Based on the image, describe why these players are or are not in offside positions. "
                f"Be concise (2-3 sentences)."
            )

            if QWEN_UTILS:
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_img},
                        {"type": "text", "text": prompt},
                    ]
                }]
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, _ = process_vision_info(messages)
                inputs = self.processor(
                    text=[text], images=image_inputs, return_tensors="pt"
                ).to(self.model.device)
            else:
                inputs = self.processor(
                    text=prompt, images=pil_img, return_tensors="pt"
                ).to(self.model.device)

            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=150)
            input_len = inputs["input_ids"].shape[1]
            qwen_output = self.processor.decode(
                out[0][input_len:], skip_special_tokens=True
            ).strip()

            return qwen_output

        except Exception as e:
            print(f"[Reasoner] 추론 실패: {e}")
            return summary

    def _build_summary(self, offside_players, onside_players, team_attacking, goal_side):
        lines = [f"공격 팀: {team_attacking} | 골대 방향: {goal_side}"]
        if offside_players:
            lines.append(f"오프사이드 선수 {len(offside_players)}명 감지:")
            for p in offside_players:
                lines.append(f"  - 발 위치: {p['forward_foot']}, 유니폼: {p.get('jersey_color', '?')}")
        else:
            lines.append("오프사이드 선수 없음 (온사이드)")
        return "\n".join(lines)
