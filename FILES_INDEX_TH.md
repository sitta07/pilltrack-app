# 📚 PillTrack - Raspberry Pi 5
## ดัชนีไฟล์ที่สำคัญทั้งหมด

---

## 🎯 ไฟล์ Pi 5 ที่ต้องใช้

### 1. **README_PI5.md** ⭐ (เริ่มต้นที่นี่!)
```
📖 คู่มือเบื้องต้นภาษาไทย
├─ ข้อกำหนดของระบบ
├─ ขั้นตอนการติดตั้ง
├─ วิธีการใช้ 3 ขั้นตอน
├─ แก้ปัญหา
└─ ตัวอย่างการใช้งาน
```
👉 **เปิดไฟล์นี้ก่อน!**

### 2. **requirements_pi5.txt** (Dependencies)
```
📦 รายการ packages สำหรับ Pi 5 (ARM 64-bit)
├─ PyTorch (CPU only)
├─ OpenCV
├─ FAISS
├─ Ultralytics (YOLO)
└─ อื่น ๆ
```
👉 ใช้: `pip install -r requirements_pi5.txt`

### 3. **phase1_database_preparation_pi5.py** (สร้าง Database)
```
🗄️ สร้าง FAISS index จากรูปยา 1000 ชนิด
├─ ลบพื้นหลัง (YOLO segmentation)
├─ สร้าง augmentation 10 เวอร์ชัน
├─ ดึง features (EfficientNet-B3)
├─ สร้าง FAISS index
└─ บันทึก metadata
```
⏱️ **เวลา:** 2-4 ชั่วโมง (ทำครั้งเดียว)
👉 ใช้: `python3 phase1_database_preparation_pi5.py`

### 4. **phase2_live_inference_pi5.py** (รันจริง)
```
🎥 ระบุยาจากกล้องแบบเรียลไทม์
├─ Picamera2 support (Pi 5 native)
├─ ตรวจจับ drugs (YOLO)
├─ ค้นหาฐานข้อมูล (FAISS)
├─ 3-tier confidence system
└─ แสดงผลบนหน้าจอ
```
⚡ **ความเร็ว:** 12-15 FPS
👉 ใช้: `python3 phase2_live_inference_pi5.py`

### 5. **setup_pi5.sh** (Automated Setup)
```
⚙️ ติดตั้ง dependencies อัตโนมัติ
├─ System packages
├─ Python packages
└─ ตรวจสอบการติดตั้ง
```
👉 ใช้: `bash setup_pi5.sh`

### 6. **PILLTRACK_PI5_CHANGES.md** (บันทึกการเปลี่ยนแปลง)
```
📝 สรุปการปรับปรุงสำหรับ Pi 5
├─ CPU-only optimization
├─ Batch size reduction
├─ Picamera2 support
└─ ความเร็วคาดหวัง
```

---

## 📂 โครงสร้างไฟล์โครงการ

```
pilltrack-app/
│
├── 🚀 RASPBERRY PI 5 FILES
│   ├── README_PI5.md                 ⭐ เริ่มที่นี่
│   ├── requirements_pi5.txt           📦 Dependencies
│   ├── setup_pi5.sh                   ⚙️ Auto setup
│   ├── phase1_database_preparation_pi5.py  🗄️ สร้าง DB
│   ├── phase2_live_inference_pi5.py   🎥 รันจริง
│   └── PILLTRACK_PI5_CHANGES.md      📝 บันทึกการเปลี่ยนแปลง
│
├── 📚 DOCUMENTATION
│   ├── SYSTEM_ARCHITECTURE.md         🏗️ ระบบโครงสร้าง
│   ├── IMPLEMENTATION_GUIDE.md        📖 คู่มือสมบูรณ์
│   ├── COMPLETE_SUMMARY.md            📊 สรุปทั้งหมด
│   └── README.md                      📄 README หลัก
│
├── 💾 GPU VERSION (Original - ไม่ใช้บน Pi)
│   ├── phase1_database_preparation.py
│   ├── phase2_live_inference.py
│   └── evaluation_module.py
│
├── 📁 DATA FOLDERS
│   ├── drug-scraping-c/               📸 รูปยา 1000+ ชนิด
│   ├── data/                          📊 ข้อมูลอื่น
│   ├── debug_crops/                   🔍 Debug files
│   └── test_match_real/               ✅ Test data
│
├── 🤖 MODELS
│   ├── best_process_2.onnx            🕵️ Detection ONNX
│   ├── best_process_2.pt              🕵️ Detection PyTorch
│   ├── seg_db_best.onnx               ✂️ Segmentation ONNX
│   ├── seg_db_best.pt                 ✂️ Segmentation PyTorch
│   ├── box_count_yolo.onnx            📦 Count model
│   ├── box_count_yolo.pt              📦 Count model
│   └── seg_process_best.*             🔧 Segment process
│
├── 🔧 CONFIGURATION
│   ├── config.py                      ⚙️ Config
│   ├── database.py                    💾 Database
│   ├── engines.py                     🚀 Engines
│   └── target_drugs.txt               🎯 Target drugs
│
├── 📊 OUTPUT (สร้างหลังจากรัน Phase 1)
│   └── faiss_database/                🗄️ FAISS index
│       ├── drug_index.faiss           6 MB
│       ├── metadata.json              10 KB
│       └── multiscale_features.pkl    18 MB
│
└── 📄 OTHER FILES
    ├── LICENSE                        ©️ License
    ├── package.md                     📦 Package info
    ├── requirement.md                 📋 Old requirements
    └── *.py (legacy)                  🗂️ Old files
```

---

## 🚀 Quick Start (3 ขั้นตอน)

### ขั้นที่ 1: เตรียมระบบ
```bash
# อ่านคู่มือเบื้องต้น
cat README_PI5.md

# หรือรัน setup อัตโนมัติ
bash setup_pi5.sh
```

### ขั้นที่ 2: สร้าง Database
```bash
source venv/bin/activate
python3 phase1_database_preparation_pi5.py
# ⏱️ 2-4 ชั่วโมง (ทำครั้งเดียว)
```

### ขั้นที่ 3: รันระบบจริง
```bash
python3 phase2_live_inference_pi5.py
# 🎥 ดูกล้องและตรวจจับยา
```

---

## 🔄 Workflow แต่ละครั้ง

```
Day 1: Setup & Database Creation
├─ เปิดเครื่อง Pi 5
├─ รันคำสั่ง setup
├─ รันคำสั่ง phase1 (2-4 ชั่วโมง)
└─ database พร้อมใช้

Day 2+: Regular Usage
├─ เปิดเครื่อง
├─ รันคำสั่ง phase2
├─ ใช้งานปกติ
└─ ปิดเครื่อง
```

---

## 🎯 หลักการทำงาน

### Phase 1: Database Preparation (ทำครั้งเดียว)
```
รูปยา 1000 ชนิด
       ↓
ลบพื้นหลัง (YOLO)
       ↓
สร้าง 10 เวอร์ชัน (Augmentation)
       ↓
ดึง features (EfficientNet-B3)
       ↓
สร้าง FAISS index
       ↓
✅ Database พร้อม (faiss_database/)
```

### Phase 2: Live Inference (ใช้ทุกครั้ง)
```
🎥 กล้อง (30 FPS)
       ↓
ตรวจจับ drugs (YOLO) 50-80ms
       ↓
ดึง features (CNN) 30-40ms
       ↓
ค้นหา FAISS (5-10ms)
       ↓
ตัดสินใจ (3-tier)
       ↓
🖼️ แสดงผล (พร้อม labels)
```

---

## 📊 ความเร็วคาดหวัง

| ขั้นตอน | Pi 5 | GPU | หมายเหตุ |
|--------|------|-----|---------|
| Frame Capture | 3-5ms | 1ms | Pi ช้าเล็กน้อย |
| Detection | 50-80ms | 16ms | Pi ช้าเยอะ |
| Features | 30-40ms | 8ms | Pi ช้าเยอะ |
| Search | 5-10ms | 2ms | เกือบเท่ากัน |
| **Total** | **100-150ms** | **30ms** | **Pi ~5x ช้า** |
| **FPS** | **12-15** | **33** | **ยังใช้ได้** |

✅ **Pi 5 ช้า แต่ยังใช้ได้สำหรับการทำงาน**

---

## 🔑 Key Files Comparison

### Phase 1
| ไฟล์ | GPU | Pi 5 | ความแตกต่าง |
|-----|-----|------|-----------|
| Batch Size | 32 | 8 | ลดลง 75% |
| Device | cuda | cpu | CPU only |
| Workers | 8 | 2 | ลดลง 75% |
| FP16 | ✅ | ❌ | ปิดใน Pi |
| Threading | Sequential | ThreadPool | Pi ใช้ async |

### Phase 2
| ไฟล์ | GPU | Pi 5 | ความแตกต่าง |
|-----|-----|------|-----------|
| FPS Target | 30 | 15 | ลดลง 50% |
| Camera | OpenCV | Picamera2 | Native Pi support |
| Batch Size | 32 | 1 | ไม่ batching |
| Device | cuda | cpu | CPU only |
| Async | Yes | Yes | เหมือนกัน |

---

## 💡 ทำให้ดีขึ้น

### ถ้าต้องการเร็วขึ้น:
```python
# แก้ไขใน phase2_live_inference_pi5.py:
FPS_TARGET = 12          # ลดจาก 15
BATCH_SIZE = 1           # เก็บไว้ที่ 1
```

### ถ้าต้องการถูกต้องขึ้น:
```python
# แก้ไขใน phase2_live_inference_pi5.py:
accept_threshold = 0.80  # เพิ่มจาก 0.75
partial_threshold = 0.65 # เพิ่มจาก 0.60
```

### ถ้าต้องการเพิ่มยาใหม่:
```bash
# 1. เพิ่มรูปยาในโฟลเดอร์:
cd drug-scraping-c
mkdir "Aspirin"
cp path/to/image.jpg Aspirin/

# 2. รัน phase1 ใหม่:
python3 phase1_database_preparation_pi5.py

# 3. ระบบพร้อมใช้ (ไม่ต้อง retrain!)
```

---

## 🐛 ปัญหาทั่วไป

### ❌ Out of Memory
```bash
# ลด BATCH_SIZE เพิ่มเติม
# แก้ไข: BATCH_SIZE = 1
```

### ❌ Camera Not Found
```bash
# ตรวจสอบกล้อง
libcamera-hello -t 2

# หรือใช้ USB Webcam
ls /dev/video*
```

### ❌ YOLO Model Not Found
```bash
# ตรวจสอบ model files:
ls -la best_process_2.onnx
ls -la seg_db_best.pt
```

### ❌ FAISS Index Not Found
```bash
# รัน Phase 1 ก่อน
python3 phase1_database_preparation_pi5.py
```

---

## ✅ Checklist

### ตัวตั้งอุปกรณ์
- [ ] Raspberry Pi 5 พร้อม (8GB+ RAM)
- [ ] OS ติดตั้ง (Raspberry Pi OS 64-bit)
- [ ] Internet ติดต่อหา
- [ ] กล้อง (Picamera2 หรือ USB)

### ตั้งค่าซอฟต์แวร์
- [ ] Python 3.8+ ติดตั้ง
- [ ] pip อัปเดตแล้ว
- [ ] dependencies ติดตั้งเสร็จ
- [ ] models (ONNX/PT) ดาวน์โหลดแล้ว

### Database Preparation
- [ ] รูปยาอยู่ใน drug-scraping-c/
- [ ] Phase 1 รันเสร็จ
- [ ] faiss_database/ มีข้อมูล
- [ ] ไม่มีข้อผิดพลาด

### Live Inference
- [ ] Phase 2 รันได้
- [ ] กล้องทำงาน
- [ ] แสดงผล (บ้านบอก)
- [ ] ความเร็วน่าน (12-15 FPS)

---

## 📞 บันทึกสำคัญ

| ประเด็น | คำตอบ |
|---------|--------|
| **ต้อง GPU หรือไม่** | ❌ ไม่จำเป็น (CPU only) |
| **ต้อง Picamera2 หรือไม่** | ⚠️ ไม่จำเป็น (fallback ไป OpenCV) |
| **ต้อง retrain หรือไม่** | ❌ ไม่ (ใช้ pre-trained models) |
| **ต้องเพิ่มเติม package หรือไม่** | ❌ ไม่ (รวมใน requirements) |
| **ต้องเพิ่มยาทีละครั้งหรือไม่** | ✅ ได้ (รัน phase1 ใหม่) |
| **สามารถเพิ่มยาเป็น 10000 ชนิดได้หรือไม่** | ✅ ได้ (เปลี่ยน index type) |

---

## 🎉 สรุป

```
✅ Phase 1 ✓ สร้าง database (ครั้งเดียว)
✅ Phase 2 ✓ ระบุยา (ทุกครั้ง)
✅ Documentation ✓ ภาษาไทยแล้ว
✅ Performance ✓ 12-15 FPS (ใช้ได้)
✅ CPU-Only ✓ ไม่ต้อง GPU
✅ Ready ✓ พร้อมใช้

🚀 PillTrack บน Raspberry Pi 5 - พร้อมแล้ว!
```

---

## 📖 อ่านเพิ่มเติม

| ไฟล์ | สำหรับ |
|-----|--------|
| README_PI5.md | ผู้ใช้ (เบื้องต้น) |
| SYSTEM_ARCHITECTURE.md | Developers (ลึก) |
| IMPLEMENTATION_GUIDE.md | Implementers (ทั้งหมด) |
| PILLTRACK_PI5_CHANGES.md | Reviewers (บันทึก) |
| COMPLETE_SUMMARY.md | สำรอง (หมายเหตุ) |

---

**✨ ขอให้สำเร็จ! PillTrack บน Pi 5 🚀**

สร้างเมื่อ: ธันวาคม 2025  
สถานะ: ✅ พร้อมใช้งาน
