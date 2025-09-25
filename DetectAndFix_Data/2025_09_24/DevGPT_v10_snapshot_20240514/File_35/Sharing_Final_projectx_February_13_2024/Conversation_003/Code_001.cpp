void checkSwitches() {
  static unsigned long lastSW4Press = 0;
  static bool servoActivated = false; // เพิ่มตัวแปรสถานะเพื่อตรวจสอบว่าเซอร์โวถูกเปิดหรือไม่
  static unsigned long buzzerStartTime = 0; // เพิ่มตัวแปรเพื่อเก็บเวลาเริ่มต้นของเสียงบัสเซอร์
  unsigned long currentMillis = millis(); // เวลาปัจจุบัน

  if (digitalRead(sw1) == LOW) {
    Serial.println("SW1 pressed");
  }
  if (digitalRead(sw2) == LOW) {
    Serial.println("SW2 pressed");
    myservo.write(180);
    servoActivated = true; // เซอร์โวถูกเปิด
    buzzerStartTime = currentMillis; // เริ่มต้นเวลาสำหรับเสียงบัสเซอร์
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
      buzzerStartTime = currentMillis; // เริ่มต้นเวลาสำหรับเสียงบัสเซอร์
    }
    else if (currentMillis - lastSW4Press >= 3000 && servoActivated) { // ตรวจสอบว่าผ่านไปเวลา 3 วินาทีและเซอร์โวถูกเปิด
      myservo.write(0);
      lastSW4Press = currentMillis; // อัปเดตเวลาที่ SW4 ถูกกดล่าสุด
      servoActivated = false; // เซอร์โวถูกปิด
    }
  }
  
  // เช็คเวลาที่เซอร์โวทำงานที่ 180 องศาและเปิด buzzer 0.5 วินาที
  if (servoActivated && myservo.read() == 180 && currentMillis - buzzerStartTime <= 500) {
    digitalWrite(buzzer, HIGH); // เปิด buzzer
  } else {
    digitalWrite(buzzer, LOW); // ปิด buzzer
  }
}