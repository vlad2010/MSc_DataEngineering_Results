void callback(char* topic, byte* payload, unsigned int length) {
  // ตรวจสอบว่าข้อความที่รับมาเป็นข้อมูลเวลาหรือไม่
  if (strcmp(topic, "@msg/timer/setHr") == 0) {
    receivedHour = atoi((char *)payload);
  } else if (strcmp(topic, "@msg/timer/setMinute") == 0) {
    receivedMinute = atoi((char *)payload);
  } else if (strcmp(topic, "@msg/timer/setSec") == 0) {
    receivedSecond = atoi((char *)payload);
  }
}