import json
import os


import cv2
import numpy as np


class Config:
    def __init__(self):
        self.threshold = 0.80
        self.f_cooldown = 0.60
        self.f_jitter = 0.30
        self.input_method = 0
        self.auto_press_enabled = True
        self.deadzone_ratio = 0.30
        self.catch_count = 0
        self.center_deadzone_ratio = 0.08
        self.prediction_time = 0.08
        self.marker_hsv_low = [20, 100, 100]
        self.marker_hsv_high = [35, 255, 255]
        self.marker_hsv_trained = False
        self.marker_samples_hsv = []

    def load(self, path="config.json"):
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            self.threshold = cfg.get("threshold", self.threshold)
            self.f_cooldown = cfg.get("f_cooldown", self.f_cooldown)
            self.f_jitter = cfg.get("f_jitter", self.f_jitter)
            self.input_method = cfg.get("input_method", self.input_method)
            self.auto_press_enabled = cfg.get("auto_press_enabled", self.auto_press_enabled)
            self.deadzone_ratio = cfg.get("deadzone_ratio", self.deadzone_ratio)
            self.catch_count = cfg.get("catch_count", self.catch_count)
            self.center_deadzone_ratio = cfg.get("center_deadzone_ratio", self.center_deadzone_ratio)
            self.prediction_time = cfg.get("prediction_time", self.prediction_time)
            self.marker_hsv_low = cfg.get("marker_hsv_low", self.marker_hsv_low)
            self.marker_hsv_high = cfg.get("marker_hsv_high", self.marker_hsv_high)
            self.marker_hsv_trained = cfg.get("marker_hsv_trained", self.marker_hsv_trained)
            self.marker_samples_hsv = cfg.get("marker_samples_hsv", self.marker_samples_hsv)
        except Exception as e:
            print(f"[WARN] Config: {e}")

    def save(self, path="config.json"):
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"[ERROR] Save config: {e}")
            return False

    def to_dict(self):
        return {
            "threshold": self.threshold,
            "f_cooldown": self.f_cooldown,
            "f_jitter": self.f_jitter,
            "input_method": self.input_method,
            "auto_press_enabled": self.auto_press_enabled,
            "deadzone_ratio": self.deadzone_ratio,
            "catch_count": self.catch_count,
            "center_deadzone_ratio": self.center_deadzone_ratio,
            "prediction_time": self.prediction_time,
            "marker_hsv_low": self.marker_hsv_low,
            "marker_hsv_high": self.marker_hsv_high,
            "marker_hsv_trained": self.marker_hsv_trained,
            "marker_samples_hsv": self.marker_samples_hsv,
        }

    # ── marker training helpers ──────────────────────────────────────

    def add_marker_sample(self, hv, sv, vv):
        self.marker_samples_hsv.append({
            "hv": hv,
            "sv": sv,
            "vv": vv,
        })

    def compute_hsv_from_samples(self):
        if not self.marker_samples_hsv:
            return False
        try:
            all_h = []
            all_s = []
            all_v = []
            for s in self.marker_samples_hsv:
                all_h.extend(s["hv"])
                all_s.extend(s["sv"])
                all_v.extend(s["vv"])
            if not all_h:
                return False
            low_h = max(0, int(np.percentile(all_h, 5)))
            high_h = min(179, int(np.percentile(all_h, 95)))
            low_s = max(0, int(np.percentile(all_s, 10)))
            low_v = max(0, int(np.percentile(all_v, 10)))
            self.marker_hsv_low = [low_h, low_s, low_v]
            self.marker_hsv_high = [high_h, 255, 255]
            self.marker_hsv_trained = True
            return True
        except Exception as e:
            print(f"[WARN] compute_hsv: {e}")
            return False

    def clear_marker_training(self):
        self.marker_samples_hsv = []
        self.marker_hsv_low = [20, 100, 100]
        self.marker_hsv_high = [35, 255, 255]
        self.marker_hsv_trained = False

    def calibrate_hsv_from_image(self, img):
        try:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            valid = (hsv[:, :, 1] > 30) & (hsv[:, :, 2] > 30)
            hv = hsv[:, :, 0][valid]
            sv = hsv[:, :, 1][valid]
            vv = hsv[:, :, 2][valid]
            if hv.size < 5:
                return False
            self.add_marker_sample(hv.tolist(), sv.tolist(), vv.tolist())
            return True
        except Exception as e:
            print(f"[WARN] calibrate_hsv: {e}")
            return False
