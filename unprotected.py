import os
import sys
import cv2
import time
import torch
import argparse
import numpy as np
import matplotlib
import yolov5
import math
matplotlib.use('Agg')
from pathlib import Path
from collections import Counter
import torch.backends.cudnn as cudnn
from utils.general import set_logging
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, 
                            check_imshow, check_requirements, colorstr, cv2,
                            increment_path, non_max_suppression, print_args,
                            scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

# Import the necessary libraries for the calc_distances function
import cv2
import math

# Define the calc_distances function
def calc_distances(results, frame):
    cord, scores, labels = results
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    points = []
    distances = []  # Create a list to store distances

    for car in cord:
        x1, y1, x2, y2 = int(car[0] * x_shape), int(car[1] * y_shape), int(car[2] * x_shape), int(
            car[3] * y_shape)
        x_mid_rect, y_mid_rect = (x1 + x2) / 2, (y1 + y2) / 2
        y_line_length, x_line_length = abs(y1 - y2), abs(x1 - x2)
        points.append([x1, y1, x2, y2, int(x_mid_rect), int(y_mid_rect), int(x_line_length), int(y_line_length)])

    x_shape_mid = int(x_shape / 2)
    start_x, start_y = x_shape_mid, y_shape
    start_point = (start_x, start_y)

    heigth_in_rf = 121
    measured_distance = 275
    real_heigth = 60
    focal_length = (heigth_in_rf * measured_distance) / real_heigth

    pixel_per_cm = float(2200 / x_shape) * 2.54
    for i in range(0, len(points)):
        end_x1, end_y1, end_x2, end_y2, end_x_mid_rect, end_y_mid_rect, end_x_line_length, end_y_line_length = points[i]
        if end_x2 < x_shape_mid:
            end_point = (end_x2, end_y2)
        elif end_x1 > x_shape_mid:
            end_point = (end_x1, end_y2)
        else:
            end_point = (end_x_mid_rect, end_y2)

        dif_x, dif_y = abs(start_point[0] - end_point[0]), abs(start_point[1] - end_point[1])
        pixel_count = math.sqrt(math.pow(dif_x, 2) + math.pow(dif_y, 2))

        distance = float(pixel_count * pixel_per_cm / end_y_line_length)
        distances.append(distance)  # Append the calculated distance to the list

    return distances  # Return the list of distances

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] 
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT)) 
ROOT = Path(os.path.relpath(ROOT, Path.cwd())) 

#---------------Object Tracking---------------
import skimage
from sort import *

#-----------Object Blurring-------------------
blurratio = 40

#.................. Tracker Functions .................
'''Computer Color for every box and track'''
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
def compute_color_for_labels(label):
    color = [int(int(p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

"""" Calculates the relative bounding box from absolute pixel values. """
def bbox_rel(*xyxy):
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

# 이전 프레임의 거리 및 주행상태를 저장하는 변수를 추가합니다.
prev_distances = [0] * 40
prev_texts = ["unknown"] * 40
prev_distance_idx = 0  # 이전 프레임의 거리 및 주행상태를 업데이트하기 위한 인덱스

prev_detected_classes = []

def draw_boxes(img, bbox, identities=None, categories=None, names=None, color_box=None, offset=(0, 0)):
    global prev_distances, prev_texts, prev_distance_idx

    distances = calc_distances((bbox, [], []), img)  # Calculate distances
    prev_text = "unknown"  # Initialize previous text as "unknown"
    prev_distance = 0  # Initialize previous distance as 0
    min_opposite_distance = float('inf')  # 초기값을 무한대로 설정
    detected_classes = []
    for i, (box, distance) in enumerate(zip(bbox, distances)):  # Iterate over distances and detections
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        data = (int((box[0] + box[2]) / 2), (int((box[1] + box[3]) / 2)))
        label = str(id)

        

        

         # Display class names if available
        if names is not None:
            class_name = names[cat] if cat < len(names) else 'N/A'
            cv2.putText(img, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            if class_name in ['plate', 'go', 'left go','stop','car']:
                detected_classes.append(class_name)

        if color_box:
            color = compute_color_for_labels(id)
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        if categories[i] in [0, 1, 2, 3]:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if names[cat] == 'car':
            cv2.putText(img, str(round(distance, 2)) + " m", (int(x1), int(y2)), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 2)  # Display distance

        distance_change = abs(distance - prev_distance)

        # Classify vehicle status based on position in the frame
        if categories[i] == 4:
            frame_midpoint = img.shape[1] // 2 - 20  # 1/3 point from the left
            if data[0] < frame_midpoint:  # If the midpoint of the vehicle is on the left side
                text = "opposite"
                min_opposite_distance = min(min_opposite_distance, distance)
            else:  # If the midpoint of the vehicle is on the right side
                text = "forward"

            # Choose text color based on object status
            if text == "opposite":
                text_color = (0, 0, 255)  # Red for "opposite"
            else:
                text_color = (255, 0, 0)  # Blue for "forward"

            if color_box:
                color = compute_color_for_labels(id)
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                # Display 'text' inside the bounding box only if it's not "unknown"
                if text != "unknown":
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                    text_x = x1 + (x2 - x1 - text_size[0]) // 2  # Center text horizontally
                    text_y = y1 + (y2 - y1 + text_size[1]) // 2  # Center text vertically
                    cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

                cv2.circle(img, data, 3, color, -1)
                

            else:
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 191, 0), 2)

                # Display 'text' inside the bounding box only if it's not "unknown"
                if text != "unknown":
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                    text_x = x1 + (x2 - x1 - text_size[0]) // 2  # Center text horizontally
                    text_y = y1 + (y2 - y1 + text_size[1]) // 2  # Center text vertically
                    cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

                cv2.circle(img, data, 3, (255, 191, 0), -1)
        
    
        
        print(detected_classes)
        if 'plate' in detected_classes and 'stop' in detected_classes: 
            # stop.jpg 이미지를 영상 상단 왼쪽 출력
            print("무조건 멈춰")
            img = insert_images(img, ["./stop.jpg"], x_offset=1150, y_offset=100)

        elif 'plate' in detected_classes and 'go' in detected_classes and min_opposite_distance <= 50:
            # stop.jpg 이미지를 영상 상단 왼쪽 출력
            print("50미터 이하니까 출발 금지")
            time.sleep(0.1)
            img = insert_images(img, ["./stop.jpg"], x_offset=1150, y_offset=100)

        elif 'plate' in detected_classes and 'left go' in detected_classes : 
            # go.jpg 이미지를 영상 상단 왼쪽 출력
            print("무조건 출발")
            img = insert_images(img, ["./go.jpg"], x_offset=1150, y_offset=100)
            time.sleep(0.1)

        elif 'plate' in detected_classes and 'go' in detected_classes and min_opposite_distance >= 50:
            # go.jpg 이미지를 영상 상단 왼쪽에 출력
            print("50미터 이상이니까 출발 가능")
            img = insert_images(img, ["./go.jpg"], x_offset=1150, y_offset=100)
            time.sleep(0.1)
        
    return img


@torch.no_grad()
def detect(weights=ROOT / '/home/ubuntu/yolov5/deep.pt',
        source=ROOT / '/home/ubuntu/rodeview',
        data=ROOT / '/home/ubuntu/yolov5/data/adas.yaml',
        imgsz=(640, 640),conf_thres=0.25,iou_thres=0.45,
        max_det=1000, device='cpu',  view_img=False,
        save_txt=False, save_conf=False, save_crop=False,
        nosave=False, classes=None,  agnostic_nms=False,
        augment=False, visualize=False,  update=False,
        project=ROOT / 'runs/detect',  name='exp',
        exist_ok=False, line_thickness=2,hide_labels=False,
        hide_conf=False,half=False,dnn=False,display_labels=False,
        blur_obj=False,color_box = False,):

    save_img = not nosave and not str(source).endswith('.txt')    


    #.... Initialize SORT ....
    sort_max_age = 5
    sort_min_hits = 2
    sort_iou_thresh = 0.2
    sort_tracker = Sort(max_age=sort_max_age,
                       min_hits=sort_min_hits,
                       iou_threshold=sort_iou_thresh)
    track_color_id = 0
    #.........................
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'

    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)

    half &= (pt or jit or onnx or engine) and device.type != 'cpu'
    if pt or jit:
        model.model.half() if half else model.model.float()

    if webcam:
        cudnn.benchmark = True
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1
    vid_path, vid_writer = [None] * bs, [None] * bs

    t0 = time.time()

    dt, seen = [0.0, 0.0, 0.0], 0

    for path, im, im0s, vid_cap, s in dataset:

        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            imc = im0.copy() if save_crop else im0
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                # Call calc_distances function here to display distances
                calc_distances((det[:, :4], det[:, 4], det[:, 5]), im0)

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if blur_obj:
                        crop_obj = im0[int(xyxy[1]):int(xyxy[3]),int(xyxy[0]):int(xyxy[2])]
                        blur = cv2.blur(crop_obj,(blurratio,blurratio))
                        im0[int(xyxy[1]):int(xyxy[3]),int(xyxy[0]):int(xyxy[2])] = blur
                    else:
                        continue
                #..................USE TRACK FUNCTION....................
                #pass an empty array to sort
                dets_to_sort = np.empty((0,6))

                # NOTE: We send in detected object class too
                for x1,y1,x2,y2,conf,detclass in det.cpu().detach().numpy():
                    dets_to_sort = np.vstack((dets_to_sort,
                                              np.array([x1, y1, x2, y2,
                                                        conf, detclass])))

                # Run SORT
                tracked_dets = sort_tracker.update(dets_to_sort)
                tracks =sort_tracker.getTrackers()

                      
                # draw boxes for visualization
                if len(tracked_dets)>0:
                    bbox_xyxy = tracked_dets[:,:4]
                    identities = tracked_dets[:, 8]
                    categories = tracked_dets[:, 4]
                    draw_boxes(im0, bbox_xyxy, identities, categories, names,color_box)






            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()
                        if vid_cap:
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)
        print("Frame Processing!")
    print("Video Exported Success")

    if update:
        strip_optimizer(weights)

    if vid_cap:
        vid_cap.release()


def insert_images(img, image_paths, x_offset=0, y_offset=0):
    for image_path in image_paths:
        # 이미지를 로드합니다
        insert_img = cv2.imread(image_path)
        
        # 이미지 크기를 조정합니다
        insert_img = cv2.resize(insert_img, (88, 112))  # 원하는 크기로 조정
        
        # 이미지를 삽입하려는 위치가 원본 이미지의 범위를 벗어나지 않도록 검사합니다
        if y_offset < img.shape[0] and x_offset < img.shape[1]:
            img[y_offset:y_offset+insert_img.shape[0], x_offset:x_offset+insert_img.shape[1]] = insert_img
        else:
            print("이미지를 삽입하려는 위치가 원본 이미지의 범위를 벗어납니다.")
    return img



def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--blur-obj', action='store_true', help='Blur Detected Objects')
    parser.add_argument('--color-box', action='store_true', help='Change color of every box and track')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt



def main(opt):
    # ... (rest of the main function remains unchanged)
    check_requirements(exclude=('tensorboard', 'thop'))

    # 검출하려는 클래스를 지정합니다 (예: car, bus, truck).
    class_indices = [0, 1, 2, 3, 4]  

    # opt.classes 인수를 수정하여 클래스 인덱스를 지정합니다.
    opt.classes = class_indices

    detect(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    print_args(vars(opt))
    main(opt)