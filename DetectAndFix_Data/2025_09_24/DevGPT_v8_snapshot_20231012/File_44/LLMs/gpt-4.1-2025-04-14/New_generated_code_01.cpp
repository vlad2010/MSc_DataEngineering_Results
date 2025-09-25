if (i2c_dev) {
  delete i2c_dev;
}
i2c_dev = new Adafruit_I2CDevice(addr, theWire);