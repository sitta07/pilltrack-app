import cv2

print("🔍 Scanning for cameras...")
found = False

# ลองเช็ค 0 ถึง 4
for index in range(5):
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"✅ Camera found at index: {index} ({frame.shape[1]}x{frame.shape[0]})")
            found = True
        cap.release()

if not found:
    print("❌ No cameras found!")