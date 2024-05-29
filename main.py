import copy
import logging
import os
from typing import List

import cv2
import numpy as np

# Tracking
import norfair
from norfair import Tracker, Detection
from norfair.tracker import TrackedObject

from src.HandyClasses import ConVideoCapture
from src.HandyFunctions import softmax
from src.YoloDetection import YoloDetection

# Required confidence before confirming something is an object (max 1.0)
CONFIDENCE_THRESH: float = 0.1
DIST_THRESH_BBOX: float = 0.7
NMS_THRESHOLD: float = 0.25
# idx 0 -> human
PERSON_IDX: int = 0
SAVE_ROIS: bool = False

# TODO: Error handling
# TODO: Unit tests

if __name__ == "__main__":
    # TODO: Cite footage and yolo
    # TODO: Add argparser

    video_path = os.path.join(".", "data", "mall-2.mp4")
    model_path = os.path.join(".", "models", "yolov9-c-converted.onnx")
    gender_model_path = os.path.join(".", "models", "gender_classifier.onnx")

    net = cv2.dnn.readNetFromONNX(model_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_VKCOM)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_VULKAN)

    gender_net = cv2.dnn.readNetFromONNX(gender_model_path)
    gender_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_VKCOM)
    gender_net.setPreferableTarget(cv2.dnn.DNN_TARGET_VULKAN)

    tracker = Tracker(distance_function="iou", distance_threshold = DIST_THRESH_BBOX)

    image_count = 4774
    num_crossings_women = 0
    num_crossings_men = 0

    with ConVideoCapture(video_path) as video_stream:

        while video_stream.isOpened():
            is_read, frame = video_stream.read()

            if not is_read:
                break

            draw_frame: np.ndarray = frame.copy()
            cvt_frame = frame.astype(np.float32)
            blob = cv2.dnn.blobFromImage(cvt_frame, 1.0/255.0, size=(640,640), swapRB=True, crop=False)
            resize_x = float(cvt_frame.shape[1] / blob.shape[3])
            resize_y = float(cvt_frame.shape[0] / blob.shape[2])
            # prediction = session.run(None, {input_name: blob})[0]
            net.setInput(blob)
            prediction = net.forward()

            # Model format: 1x84x8600
            # 4 rows for x1, y1, x2, y2 -> 80 rows for classes
            # 8600 columns for anchor boxes

            prediction_pos = prediction[0, 0:4, :]
            prediction_confidence = prediction[0, 4:, :]

            norfair_detections: List[Detection] = []
            yolo_detections: List[YoloDetection] = []

            for col_num in range(prediction_pos.shape[1]):
                person_confidence = float(prediction_confidence[PERSON_IDX, col_num])
                detect = YoloDetection(person_confidence,
                                       pred_positions = prediction[0, 0:4, col_num],
                                       resize=(resize_x, resize_y))
                yolo_detections.append(detect)

            # NMS
            bboxes = [(x.tlx, x.tly, x.brx, x.bry) for x in yolo_detections]
            scores = [x.confidence for x in yolo_detections]
            (nms_confidences, nms_indices) = cv2.dnn.softNMSBoxes(bboxes=bboxes,
                                 scores=scores,
                                 score_threshold=CONFIDENCE_THRESH,
                                 nms_threshold=NMS_THRESHOLD)
            updated_yolo_detections = []
            for nms_confidence, nms_index in zip(nms_confidences, nms_indices):
                updated_detection = copy.copy(yolo_detections[nms_index])
                updated_detection.set_confidence(nms_confidence)
                updated_yolo_detections.append(updated_detection)

                # For training output the shapes as frames
                # Take the right half of our video as the training set and the left half for proof
                roi = frame[updated_detection.tly:updated_detection.bry,
                         updated_detection.tlx:updated_detection.brx, :]

                if roi.shape[0] < 1 or roi.shape[1] < 1:
                    continue

                if SAVE_ROIS and updated_detection.center_x > frame.shape[1] / 2:
                    file_path = os.path.join(".", "data","people_frames", f"Person_{image_count}.png")
                    cv2.imwrite(file_path, roi)
                    image_count += 1
                else:
                    cvt_roi = roi.astype(np.float32)
                    gender_blob = cv2.dnn.blobFromImage(cvt_roi, 1.0 / 255.0, size=(128, 128),
                                                        swapRB=True, crop=False)
                    gender_net.setInput(gender_blob)
                    gender_prediction = gender_net.forward()[0]
                    gender_prediction = softmax(gender_prediction)
                    updated_detection.set_gender(gender_prediction)

                # TODO: Switch tlx to tl
                cv2.rectangle(draw_frame, (updated_detection.tlx, updated_detection.tly),
                              (updated_detection.brx, updated_detection.bry),
                              color=(0, 0, 0))
                cv2.putText(draw_frame, f"{updated_detection.gender} ({updated_detection.gender_confidence:.1%})",
                            org=(updated_detection.tlx,updated_detection.tly - 10),
                            fontFace=0, fontScale=0.5, color=(0, 0, 255.0))

            yolo_detections = updated_yolo_detections

            norfair_detections = [detect.as_norfair() for detect in yolo_detections]
            frame_half = round(frame.shape[0]/2)
            frame_width = frame.shape[1]

            tracked_objects = tracker.update(detections=norfair_detections)
            norfair.draw_boxes(draw_frame, norfair_detections)
            norfair.draw_tracked_boxes(draw_frame, tracked_objects)

            obj: TrackedObject
            for obj in tracked_objects:
                x_pos = round(obj.last_detection.points[0][0])
                y_pos = round(obj.last_detection.points[0][1])

                last_x_pos = round(obj.past_detections[-1].points[0][0])
                last_y_pos = round(obj.past_detections[-1].points[0][1])

                x_vel = x_pos - last_x_pos
                y_vel = y_pos - last_y_pos

                cv2.arrowedLine(draw_frame, (x_pos, y_pos), (x_pos + x_vel, y_pos+y_vel),
                                color=(0,0,0), thickness=1)

                if hasattr(obj, "crossed_line") and obj.crossed_line:
                    continue

                if (y_pos < frame_half < y_pos + y_vel) \
                        or (y_pos > frame_half > y_pos + y_vel):
                    if obj.last_detection.data == "woman":
                        num_crossings_women += 1
                        obj.crossed_line = True
                    elif obj.last_detection.data == "man":
                        num_crossings_men += 1
                        obj.crossed_line = True

            cv2.line(draw_frame, (0, frame_half), (frame_width, frame_half), color=(0,0,255), thickness=2)
            cv2.putText(draw_frame, f"Women: {num_crossings_women}, Men: {num_crossings_men}",
                        (5, 25), fontFace=0, fontScale=0.75, color=(0,0,255))

            cv2.imshow("Frame", draw_frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
