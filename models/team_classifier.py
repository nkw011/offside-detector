import cv2
import torch
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from transformers import BlipProcessor, BlipForQuestionAnswering


class TeamClassifier:
    def __init__(self):
        print("[TeamClassifier] BLIP VQA 로딩...")
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        self.model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"[TeamClassifier] 완료 (device: {self.device})")

    def _crop_upper(self, image_np, bbox):
        """상체 크롭 (상위 55%)"""
        x1, y1, x2, y2 = bbox
        y2c = y1 + int((y2 - y1) * 0.55)
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(image_np.shape[1], x2)
        y2c = min(image_np.shape[0], y2c)
        return image_np[y1:y2c, x1:x2]

    def get_jersey_color(self, image_np, bbox):
        """BLIP VQA — 유니폼 색상 텍스트 반환 (표시용)"""
        crop = self._crop_upper(image_np, bbox)
        if crop.size == 0:
            return "unknown"
        pil_img = Image.fromarray(crop[..., ::-1])
        question = "What color is the jersey?"
        inputs = self.processor(pil_img, question, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=8)
        return self.processor.decode(out[0], skip_special_tokens=True).lower().strip()

    def _jersey_hsv_feature(self, image_np, bbox):
        """유니폼 영역 HSV 히스토그램 → 색상 특징 벡터"""
        crop = self._crop_upper(image_np, bbox)
        if crop.size == 0:
            return np.zeros(16)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # H채널 16-bin 히스토그램 (색상이 다른 팀 구별에 핵심)
        hist = cv2.calcHist([hsv], [0], None, [16], [0, 180])
        hist = cv2.normalize(hist, hist).flatten()
        # S(채도) 평균 추가 — 흰색/검정 같은 무채색 구별
        s_mean = hsv[:, :, 1].mean() / 255.0
        v_mean = hsv[:, :, 2].mean() / 255.0
        return np.append(hist, [s_mean, v_mean])

    def _pixel_hsv_feature(self, image_np, x, y, radius=20):
        """클릭 좌표 주변 픽셀의 HSV 히스토그램 (팀 색상 기준점)"""
        h, w = image_np.shape[:2]
        x1 = max(0, x - radius); y1 = max(0, y - radius)
        x2 = min(w, x + radius); y2 = min(h, y + radius)
        patch = image_np[y1:y2, x1:x2]
        if patch.size == 0:
            return np.zeros(18)
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0], None, [16], [0, 180])
        hist = cv2.normalize(hist, hist).flatten()
        s_mean = hsv[:, :, 1].mean() / 255.0
        v_mean = hsv[:, :, 2].mean() / 255.0
        return np.append(hist, [s_mean, v_mean])

    def auto_assign_teams(self, image_np, detections, attack_xy, defend_xy):
        """
        k-means(k=2)로 정확히 두 팀으로 분류.
        attack_xy: 사용자가 클릭한 공격팀 좌표 (x, y)
        defend_xy: 사용자가 클릭한 수비팀 좌표 (x, y)
        → 클릭 위치 픽셀 색상을 기준으로 클러스터 A/B 결정 (nearest detection 불필요)
        """
        n = len(detections)
        if n == 0:
            return detections

        # 1) BLIP: 유니폼 색상 텍스트 (표시용)
        for det in detections:
            det['jersey_color'] = self.get_jersey_color(image_np, det['bbox'])

        if n == 1:
            detections[0]['team'] = 'A'
            return detections

        # 2) 각 선수 HSV 히스토그램 특징
        features = np.array([
            self._jersey_hsv_feature(image_np, det['bbox']) for det in detections
        ])

        # 3) k-means k=2 강제
        km = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = km.fit_predict(features)

        # 4) 클릭 위치 픽셀 색상 → 각 클러스터 중심과의 거리로 A/B 결정
        #    (nearest detection 탐색 없이 직접 색상 비교)
        feat_a = self._pixel_hsv_feature(image_np, attack_xy[0], attack_xy[1])
        feat_b = self._pixel_hsv_feature(image_np, defend_xy[0], defend_xy[1])

        centroid_0, centroid_1 = km.cluster_centers_[0], km.cluster_centers_[1]
        dist_a0 = np.linalg.norm(feat_a - centroid_0)
        dist_a1 = np.linalg.norm(feat_a - centroid_1)
        # A 클릭 색상과 더 가까운 클러스터 = A팀
        attack_cluster = 0 if dist_a0 <= dist_a1 else 1

        # 검증: B 클릭 색상도 반대 클러스터와 더 가까워야 함
        dist_b0 = np.linalg.norm(feat_b - centroid_0)
        dist_b1 = np.linalg.norm(feat_b - centroid_1)
        defend_cluster = 0 if dist_b0 <= dist_b1 else 1
        if defend_cluster == attack_cluster:
            # 두 클릭이 같은 클러스터 → B 클릭 쪽을 반대로 강제
            attack_cluster = defend_cluster ^ 1

        for i, det in enumerate(detections):
            det['team'] = 'A' if labels[i] == attack_cluster else 'B'

        return detections
