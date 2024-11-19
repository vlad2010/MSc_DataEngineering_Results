#include <Adafruit_AD5693.h>

Adafruit_AD5693 ad5693;

void setup() {
  Serial.begin(9600);
  if (!ad5693.begin()) {
    Serial.println("Failed to initialize AD5693!");
    while (1);
  }
}

void loop() {
  // Test the library functions
  ad5693.setVoltage(1023);
 
