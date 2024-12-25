#!/usr/bin/env python3
# coding: utf-8

import cv2
import numpy as np
from cv2 import aruco

def main():
    # 1. Aruco辞書とパラメータの準備
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()

    # 2. カメラを初期化
    cap = cv2.VideoCapture(0)  # カメラデバイスID 0を使用
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    print("Press 'q' to quit the program.")

    while True:
        # 3. カメラフレームの取得
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame from the camera.")
            break

        # 4. Arucoマーカーの検出
        corners, ids, _ = aruco.detectMarkers(frame, dictionary, parameters=parameters)

        # 5. 検出結果の描画
        if ids is not None:
            print(f"Detected marker IDs: {ids.flatten()}")
            for i, corner in enumerate(corners):
                print(f"Marker ID: {ids[i][0]}, Corners: {corner[0]}")  # 各コーナー座標を表示

            # マーカーをフレームに描画
            frame = aruco.drawDetectedMarkers(frame, corners, ids)
        else:
            print("No markers detected.")

        # 6. 検出結果を表示
        cv2.imshow("Aruco Marker Detection", frame)

        # 'q'キーが押されたら終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 7. 後処理
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
