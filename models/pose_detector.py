import numpy as np
from ultralytics import YOLO

# YOLOv8-pose keypoint indices
KP_LEFT_ANKLE = 15
KP_RIGHT_ANKLE = 16
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12


class PoseDetector:
    def __init__(self, model_size="n"):
        print(f"[PoseDetector] YOLOv8{model_size}-pose 로딩...")
        self.model = YOLO(f"yolov8{model_size}-pose.pt")
        print("[PoseDetector] 완료")

    def detect(self, image, conf=0.2):
        """
        Returns list of dicts:
        {
            'bbox': (x1, y1, x2, y2),
            'conf': float,
            'keypoints': np.array shape (17, 3)  # x, y, confidence
            'left_ankle': (x, y),
            'right_ankle': (x, y),
            'forward_foot': (x, y),  # 더 앞에 있는 발목
        }
        """
        results = self.model(image, conf=conf, verbose=False)
        detections = []

        if not results or results[0].keypoints is None:
            return detections

        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        keypoints = result.keypoints.data.cpu().numpy()  # (N, 17, 3)

        for i in range(len(boxes)):
            kp = keypoints[i]  # (17, 3)
            left_ankle = kp[KP_LEFT_ANKLE][:2]
            right_ankle = kp[KP_RIGHT_ANKLE][:2]

            # 두 발목 중 신뢰도 높은 걸 사용
            la_conf = kp[KP_LEFT_ANKLE][2]
            ra_conf = kp[KP_RIGHT_ANKLE][2]

            if la_conf < 0.2 and ra_conf < 0.2:
                # 발목 미탐지 시 bbox 하단 중앙 사용
                x1, y1, x2, y2 = boxes[i]
                forward_foot = ((x1 + x2) / 2, y2)
            else:
                forward_foot = left_ankle if la_conf >= ra_conf else right_ankle

            detections.append({
                'bbox': tuple(boxes[i].astype(int)),
                'conf': float(confs[i]),
                'keypoints': kp,
                'left_ankle': tuple(left_ankle.astype(int)),
                'right_ankle': tuple(right_ankle.astype(int)),
                'forward_foot': tuple(np.array(forward_foot).astype(int)),
                'team': None,
                'is_offside': None,
            })

        return detections
