import argparse
import os
 
import cv2
import numpy as np
 
WHITE = (np.array([0, 0, 190]), np.array([179, 40, 255]))
YELLOW = (np.array([10, 110, 130]), np.array([35, 255, 255]))
EXTS = (".jpg", ".jpeg", ".png", ".bmp")
 
 
def hsv_filter(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, *WHITE) | cv2.inRange(hsv, *YELLOW)
    return mask, cv2.bitwise_and(img, img, mask=mask)
 
 
def roi(img):
    h, w = img.shape[:2]
    poly = np.array([[(0 , h), (int(.40 * w), int(.62 * h)),
                      (int(.60 * w), int(.62 * h)), (w, h)]], np.int32)
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, poly, 255)
    return cv2.bitwise_and(img, mask), poly
 
 
def process(img):
    mask, filtered = hsv_filter(img)
    mask, poly = roi(mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    blurred = cv2.GaussianBlur(mask, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    masked = edges
    result = img.copy()
    result[masked > 0] = (0, 0, 255)
    cv2.polylines(result, poly, True, (0, 255, 0), 2)
    return [("original", img), ("hsv filtered", filtered), ("blurred", blurred),
            ("canny", edges), ("roi edges", masked), ("result", result)]
 
 
def grid(stages, width=480):
    tiles = []
    for name, im in stages:
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        im = cv2.resize(im, (width, im.shape[0] * width // im.shape[1]))
        cv2.rectangle(im, (0, 0), (width, 32), (0, 0, 0), -1)
        cv2.putText(im, name, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
        tiles.append(im)
    return np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
 
 
def main():
    base = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join(base, "images"))
    ap.add_argument("--output", default=os.path.join(base, "output"))
    args = ap.parse_args()

    if os.path.isdir(args.input):
        files = [os.path.join(args.input, f) for f in sorted(os.listdir(args.input))
                 if f.lower().endswith(EXTS)]
    else:
        files = [args.input]

    os.makedirs(args.output, exist_ok=True)
    for path in files:
        img = cv2.imread(path)
        if img is None:
            print(f"[skip] {path}")
            continue
        name = os.path.splitext(os.path.basename(path))[0]
        stages = process(img)
        cv2.imwrite(f"{args.output}/{name}_result.jpg", stages[-1][1])
        cv2.imwrite(f"{args.output}/{name}_pipeline.jpg", grid(stages))
        print(f"[ok] {path}")


if __name__ == "__main__":
    main()