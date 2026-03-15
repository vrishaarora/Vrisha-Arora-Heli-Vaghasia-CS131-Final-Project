import jetson.inference
import jetson.utils
import sys

# -----------------------------
# Load DetectNet model
# -----------------------------
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.6)

# -----------------------------
# Get input/output paths
# -----------------------------
if len(sys.argv) != 3:
    print("Usage: python3 seat_detector_images.py input.jpg output.jpg")
    sys.exit(1)

input_path = sys.argv[1]
output_path = sys.argv[2]

# -----------------------------
# Load image
# -----------------------------
img = jetson.utils.loadImage(input_path)

# -----------------------------
# Run detection
# -----------------------------
detections = net.Detect(img)

chairs = []
people = []

# -----------------------------
# Separate detections
# -----------------------------
for d in detections:
    label = net.GetClassDesc(d.ClassID)

    if label == "chair":
        chairs.append(d)

    if label == "person":
        people.append(d)

# -----------------------------
# IoU function
# -----------------------------
def compute_iou(boxA, boxB):
    xA = max(boxA.Left, boxB.Left)
    yA = max(boxA.Top, boxB.Top)
    xB = min(boxA.Right, boxB.Right)
    yB = min(boxA.Bottom, boxB.Bottom)

    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight

    boxAArea = (boxA.Right - boxA.Left) * (boxA.Bottom - boxA.Top)
    boxBArea = (boxB.Right - boxB.Left) * (boxB.Bottom - boxB.Top)

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

# -----------------------------
# Determine seat status
# -----------------------------
empty_count = 0
occupied_count = 0

for chair in chairs:
    occupied = False

    for person in people:
        iou = compute_iou(chair, person)

        # Adjust threshold if needed (0.2–0.3 works well)
        if iou > 0.25:
            occupied = True
            break

    if occupied:
        occupied_count += 1
        jetson.utils.cudaDrawRect(
            img,
            (chair.Left, chair.Top, chair.Right, chair.Bottom),
            (255, 0, 0, 200)  # Red = occupied
        )
    else:
        empty_count += 1
        jetson.utils.cudaDrawRect(
            img,
            (chair.Left, chair.Top, chair.Right, chair.Bottom),
            (0, 255, 0, 200)  # Green = empty
        )

# -----------------------------
# Save output image
# -----------------------------
jetson.utils.saveImage(output_path, img)

# -----------------------------
# Print results
# -----------------------------
print("Total chairs detected:", len(chairs))
print("Empty seats:", empty_count)
print("Occupied seats:", occupied_count)
