void checkSwitches() {
  static unsigned long lastSW4Press = 0;
  static bool servoActivated = false; // เพิ่มตัวแปรสถานะเพื่อตรวจสอบว่าเซอร์โวถูกเปิดหรือไม่
  unsigned long currentMillis = millis(); // เวลาปัจจุบัน

  if (digitalRead(sw1) == LOW) {
    Serial.println("SW1 pressed");
  }
  if (digitalRead(sw2) == LOW) {
    Serial.println("SW2 pressed");
    myservo.write(180);
    servoActivated = true; // เซอร์โวถูกเปิด
  }
  if (digitalRead(sw3) == LOW) {
    Serial.println("SW3 pressed");
    myservo.write(0);
    servoActivated = false; // เซอร์โวถูกปิด
  }
  if (digitalRead(sw4) == LOW) {
    Serial.println("SW4 pressed");
    if (currentMillis - lastSW4Press >= 3000 && !servoActivated) { // ตรวจสอบว่าผ่านไปเวลา 3 วินาทีและเซอร์โวไม่ได้เปิด
      myservo.write(180);
      lastSW4Press = currentMillis; // อัปเดตเวลาที่ SW4 ถูกกดล่าสุด
      servoActivated = true; // เซอร์โวถูกเปิด
    }
    else if (currentMillis - lastSW4Press >= 3000 && servoActivated) { // ตรวจสอบว่าผ่านไปเวลา 3 วินาทีและเซอร์โวถูกเปิด
      myservo.write(0);
      lastSW4Press = currentMillis; // อัปเดตเวลาที่ SW4 ถูกกดล่าสุด
      servoActivated = false; // เซอร์โวถูกปิด
    }
  }
}