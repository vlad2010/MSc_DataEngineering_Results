void Adafruit_Debounce::begin(bool initialState) {
    // Call the original begin() function
    begin();
    
    // Set the button states to the provided initialState value
    _buttonState = initialState;
    _lastButtonState = initialState;
}
