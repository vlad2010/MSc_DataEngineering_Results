void callback(char* topic, byte* payload, unsigned int length) {
  Serial.print("Message arrived ["); // บรรทัดนี้จะแสดง "Message arrived"
  Serial.print(topic);
  Serial.print("] ");
  String message;
  for (int i = 0; i < length; i++) {
    message = message + (char)payload[i];
  }
  // print arrive msg
  if (strcmp(topic, "@msg/timer/onSet") == 0) {
    if (message == "on") {
      onSet = 1;
    } else {
      onSet = 0;
    }
  }
  if (onSet) {
    if (strcmp(topic, "@msg/timer/setHr") == 0) {
      setHour = message.toInt();
      Serial.println(setHour);
    }
    if (strcmp(topic, "@msg/timer/setMinute") == 0) {
      setMinute = message.toInt();
      Serial.println(setMinute);
    }
    if (strcmp(topic, "@msg/timer/setSec") == 0) {
      setSec = message.toInt();
      Serial.println(setSec);
    }
  }
}