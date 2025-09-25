#include <stdio.h>
#include <math.h>
#include <float.h>
#include <stdbool.h>

// Define reasonable physical limits for the exercise bike
#define MIN_CADENCE 0.0
#define MAX_CADENCE 200.0  // RPM - reasonable maximum for cycling
#define MIN_RESISTANCE 0.0
#define MAX_RESISTANCE 100.0  // Assuming percentage or level-based resistance
#define MIN_WATTAGE 0.0
#define MAX_WATTAGE 2000.0  // Professional cyclists rarely exceed this

// Error codes
typedef enum {
    CALC_SUCCESS = 0,
    CALC_ERROR_INVALID_CADENCE = -1,
    CALC_ERROR_INVALID_RESISTANCE = -2,
    CALC_ERROR_CALCULATION_OVERFLOW = -3,
    CALC_ERROR_INVALID_RESULT = -4
} CalculationError;

// Validate that a double is finite and not NaN
static bool isValidDouble(double value) {
    return isfinite(value) && !isnan(value);
}

// Secure calculation function with input validation and error handling
CalculationError calculateWattageSecure(double cadence, double resistance, double* result) {
    // Input validation - check for NULL pointer
    if (result == NULL) {
        return CALC_ERROR_INVALID_RESULT;
    }
    
    // Validate cadence input
    if (!isValidDouble(cadence) || cadence < MIN_CADENCE || cadence > MAX_CADENCE) {
        *result = 0.0;
        return CALC_ERROR_INVALID_CADENCE;
    }
    
    // Validate resistance input
    if (!isValidDouble(resistance) || resistance < MIN_RESISTANCE || resistance > MAX_RESISTANCE) {
        *result = 0.0;
        return CALC_ERROR_INVALID_RESISTANCE;
    }
    
    // Model coefficients (const for security)
    const double intercept = 12.16860795336126;
    const double coefCadence = 0.12260211;
    const double coefResistance = -0.39240546;
    const double coefCadenceSquared = 0.00464781;
    const double coefCadenceResistance = 0.34516268;
    const double coefResistanceSquared = -0.01031992;
    
    // Calculate intermediate values with overflow checking
    double cadenceSquared = cadence * cadence;
    double resistanceSquared = resistance * resistance;
    double cadenceResistanceProduct = cadence * resistance;
    
    // Check for overflow in intermediate calculations
    if (!isValidDouble(cadenceSquared) || !isValidDouble(resistanceSquared) || 
        !isValidDouble(cadenceResistanceProduct)) {
        *result = 0.0;
        return CALC_ERROR_CALCULATION_OVERFLOW;
    }
    
    // Calculate wattage with each term
    double term1 = coefCadence * cadence;
    double term2 = coefResistance * resistance;
    double term3 = coefCadenceSquared * cadenceSquared;
    double term4 = coefCadenceResistance * cadenceResistanceProduct;
    double term5 = coefResistanceSquared * resistanceSquared;
    
    // Check each term for validity
    if (!isValidDouble(term1) || !isValidDouble(term2) || !isValidDouble(term3) || 
        !isValidDouble(term4) || !isValidDouble(term5)) {
        *result = 0.0;
        return CALC_ERROR_CALCULATION_OVERFLOW;
    }
    
    // Calculate final wattage
    double wattage = intercept + term1 + term2 + term3 + term4 + term5;
    
    // Validate the final result
    if (!isValidDouble(wattage)) {
        *result = 0.0;
        return CALC_ERROR_CALCULATION_OVERFLOW;
    }
    
    // Ensure wattage is within reasonable physical bounds
    // Clamp negative values to 0 (physically impossible)
    if (wattage < MIN_WATTAGE) {
        wattage = MIN_WATTAGE;
    }
    
    // Clamp excessive values to maximum
    if (wattage > MAX_WATTAGE) {
        wattage = MAX_WATTAGE;
    }
    
    *result = wattage;
    return CALC_SUCCESS;
}

// Wrapper function for backward compatibility (with basic error handling)
double calculateWattage(double cadence, double resistance) {
    double result = 0.0;
    CalculationError error = calculateWattageSecure(cadence, resistance, &result);
    
    if (error != CALC_SUCCESS) {
        fprintf(stderr, "Warning: Calculation error occurred (code: %d)\n", error);
        return 0.0;  // Return safe default value
    }
    
    return result;
}

int main() {
    double cadence = 55;
    double resistance = 13;
    double wattage = 0.0;
    
    // Use the secure version with error handling
    CalculationError error = calculateWattageSecure(cadence, resistance, &wattage);
    
    if (error == CALC_SUCCESS) {
        printf("Predicted Wattage: %.2f Watts\n", wattage);
    } else {
        printf("Error calculating wattage. Error code: %d\n", error);
        return 1;
    }
    
    // Test edge cases for security validation
    printf("\n--- Security Test Cases ---\n");
    
    // Test negative cadence
    error = calculateWattageSecure(-10, 13, &wattage);
    printf("Negative cadence test: %s\n", error == CALC_ERROR_INVALID_CADENCE ? "PASSED" : "FAILED");
    
    // Test excessive resistance
    error = calculateWattageSecure(55, 150, &wattage);
    printf("Excessive resistance test: %s\n", error == CALC_ERROR_INVALID_RESISTANCE ? "PASSED" : "FAILED");
    
    // Test NaN input
    error = calculateWattageSecure(NAN, 13, &wattage);
    printf("NaN input test: %s\n", error == CALC_ERROR_INVALID_CADENCE ? "PASSED" : "FAILED");
    
    // Test infinity input
    error = calculateWattageSecure(55, INFINITY, &wattage);
    printf("Infinity input test: %s\n", error == CALC_ERROR_INVALID_RESISTANCE ? "PASSED" : "FAILED");
    
    return 0;
}