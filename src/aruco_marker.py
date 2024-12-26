#!/usr/bin/env python3
# coding: utf-8

import cv2
import numpy as np
from cv2 import aruco

class ArucoMarkerHandler:
    def __init__(self, dictionary_name=aruco.DICT_4X4_50):
        """
        Arucoマーカーの生成および検出を行うハンドラー。
        """
        self.dictionary = aruco.getPredefinedDictionary(dictionary_name)
        self.parameters = aruco.DetectorParameters()

    def generate_marker(self, marker_id=0, size=150, offset=10, output_file="marker.png"):
        """
        指定されたIDのArucoマーカーを生成して保存する。
        :param marker_id: 生成するマーカーのID。
        :param size: マーカーのサイズ（ピクセル単位）。
        :param offset: マーカー画像の余白（ピクセル単位）。
        :param output_file: 保存する画像ファイル名。
        :return: 生成されたマーカー画像。
        """
        # マーカー画像を生成
        ar_img = aruco.drawMarker(self.dictionary, marker_id, size)
        x_offset = y_offset = offset // 2

        # 背景を白で生成し、マーカーを中央に配置
        img = np.ones((size + offset, size + offset), dtype=np.uint8) * 255
        img[y_offset:y_offset + ar_img.shape[0], x_offset:x_offset + ar_img.shape[1]] = ar_img

        # 画像を保存
        cv2.imwrite(output_file, img)
        print(f"Generated marker saved to {output_file}")
        return img

    def detect_from_camera(self):
        """
        カメラを使用してArucoマーカーをリアルタイムで検出する。
        """
        cap = cv2.VideoCapture(0)  # カメラデバイスID 0
        if not cap.isOpened():
            print("Error: Unable to access the camera.")
            return

        print("Press 'q' to quit the program.")
        
        while True:
            # フレームの取得
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to read frame from the camera.")
                break

            # マーカー検出
            corners, ids, rejectedCandidates = aruco.detectMarkers(frame, self.dictionary, parameters=self.parameters)
            if ids is not None:
                print(f"Detected marker IDs: {ids.flatten()}")
                frame = aruco.drawDetectedMarkers(frame, corners, ids)
            else:
                print("No markers detected.")

            # 検出結果の表示
            cv2.imshow("Aruco Marker Detection", frame)

            # 'q'キーで終了
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 後処理
        cap.release()
        cv2.destroyAllWindows()

# メイン処理
if __name__ == "__main__":
    handler = ArucoMarkerHandler()

    # マーカー生成
    handler.generate_marker(marker_id=0, size=150, offset=10, output_file="marker_0.png")

    # カメラでマーカー検出
    handler.detect_from_camera()
