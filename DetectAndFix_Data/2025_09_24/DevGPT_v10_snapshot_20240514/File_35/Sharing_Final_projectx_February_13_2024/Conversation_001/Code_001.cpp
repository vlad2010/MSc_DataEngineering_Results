void checkSwitches() {
  static unsigned long lastSW4Press = 0;
  if (digitalRead(sw1) == LOW) {
    Serial.println("SW1 pressed");
  }
  if (digitalRead(sw2) == LOW) {
    Serial.println("SW2 pressed");
    myservo.write(180);
  }
  if (digitalRead(sw3) == LOW) {
    Serial.println("SW3 pressed");
    myservo.write(0);
  }
  if (digitalRead(sw4) == LOW) {
    Serial.println("SW4 pressed");
    unsigned long currentMillis = millis();
    if (currentMillis - lastSW4Press >= 3000) { // Check if 3 seconds have passed
      myservo.write(180);
      delay(3000); // Delay for 3 seconds
      myservo.write(0);
      lastSW4Press = currentMillis; // Update last press time
    }
  }
}