void checkSwitches() {
  if (digitalRead(sw1) == LOW) {
    // ใช้ค่าเวลาที่ได้รับมาจาก MQTT broker
    setHour = receivedHour;
    setMinute = receivedMinute;
    setSec = receivedSecond;
  }

  // โค้ดอื่น ๆ ในฟังก์ชัน checkSwitches()...
}