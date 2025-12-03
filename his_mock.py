class HISSystem:
    def __init__(self):
        # เพิ่ม 'name' เข้าไปในข้อมูลคนไข้
        self.mock_data = {
            "HN001": {
                "name": "Mr. Somchai Jai-dee",
                "drugs": ["duspatin_135", "orata 0.5","uroflow",'nuelin_sr_200','fah','paracap','turmeric']
            },
            "HN002": {
                "name": "Ms. Suda Rak-sa",
                "drugs": ["Lareya"]
            },
        }
    
    def get_patient_info(self, hn_id):
        """คืนค่าข้อมูลคนไข้ทั้งหมด (ชื่อ + ยา)"""
        data = self.mock_data.get(hn_id, None)
        if data:
            print(f"🏥 HIS Loaded: {data['name']} | Rx: {data['drugs']}")
            return data
        return None
    
