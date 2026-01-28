# lane_ldw.py
# Run: python lane_ldw.py --video path/to/video.mp4
import cv2
import numpy as np
import argparse

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def gaussian_blur(img, ksize=5):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

def canny(img, low=50, high=150):
    return cv2.Canny(img, low, high)

def region_of_interest(img):
    h, w = img.shape[:2]
    # focus on lower half (adjust polygon for camera)
    mask = np.zeros_like(img)
    polygon = np.array([[
        (int(0.05*w), h),
        (int(0.45*w), int(0.6*h)),
        (int(0.55*w), int(0.6*h)),
        (int(0.95*w), h),
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(img, mask)

def average_slope_intercept(lines):
    left_lines = []
    right_lines = []
    if lines is None:
        return None, None
    for l in lines:
        for x1,y1,x2,y2 in l:
            if x2==x1:
                continue
            slope = (y2-y1)/(x2-x1)
            intercept = y1 - slope*x1
            if slope < -0.3:   # left lane (tune thresholds)
                left_lines.append((slope, intercept))
            elif slope > 0.3:  # right lane
                right_lines.append((slope, intercept))
    left = np.mean(left_lines, axis=0) if left_lines else None
    right = np.mean(right_lines, axis=0) if right_lines else None
    return left, right

def make_line_points(y1, y2, line):
    if line is None:
        return None
    slope, intercept = line
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return (x1, int(y1), x2, int(y2))

def draw_lines(img, lines, color=(0,255,0), thickness=8):
    if lines is None:
        return
    for x1,y1,x2,y2 in lines:
        if x1 is None or x2 is None:
            continue
        cv2.line(img, (x1,y1), (x2,y2), color, thickness)

def process_frame(frame):
    h, w = frame.shape[:2]
    gray = grayscale(frame)
    blur = gaussian_blur(gray, 5)
    edges = canny(blur, 50, 150)
    roi = region_of_interest(edges)
    # Hough transform
    lines = cv2.HoughLinesP(roi, rho=1, theta=np.pi/180, threshold=30, minLineLength=40, maxLineGap=150)
    left, right = average_slope_intercept(lines)
    y1 = h
    y2 = int(h * 0.6)
    left_line = make_line_points(y1, y2, left)
    right_line = make_line_points(y1, y2, right)
    line_img = np.zeros_like(frame)
    draw_lines(line_img, [left_line] if left_line else [])
    draw_lines(line_img, [right_line] if right_line else [], color=(0,0,255))
    combo = cv2.addWeighted(frame, 0.8, line_img, 1.0, 0.0)
    # departure decision: check lane center vs image center
    if left_line and right_line:
        lx1,_,lx2,_ = left_line
        rx1,_,rx2,_ = right_line
        lane_x = ( (lx1+lx2)/2 + (rx1+rx2)/2 ) / 2
        center_x = w / 2
        offset_px = center_x - lane_x
        # convert px offset to simple "departure" condition (tune threshold)
        threshold_px = 0.12 * w
        status = "OK" if abs(offset_px) < threshold_px else "DEPARTURE"
        cv2.putText(combo, f"Offset px: {int(offset_px)} Status: {status}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)
    return combo

def main(args):
    cap = cv2.VideoCapture(args.video if args.video else 0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out = process_frame(frame)
        cv2.imshow("LDW", out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, help="path to video file (omit for webcam)")
    args = parser.parse_args()
    main(args)
