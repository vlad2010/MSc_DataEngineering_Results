#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <errno.h>
#include <limits.h>
#include <float.h>

#define MIN_CADENCE 0.0
#define MAX_CADENCE 200.0
#define MIN_RESISTANCE 0.0
#define MAX_RESISTANCE 100.0

double calculateWattage(double cadence, double resistance) {
    double intercept = 12.16860795336126;
    double coefCadence = 0.12260211;
    double coefResistance = -0.39240546;
    double coefCadenceSquared = 0.00464781;
    double coefCadenceResistance = 0.34516268;
    double coefResistanceSquared = -0.01031992;

    double wattage = intercept + 
                     coefCadence * cadence + 
                     coefResistance * resistance + 
                     coefCadenceSquared * cadence * cadence + 
                     coefCadenceResistance * cadence * resistance + 
                     coefResistanceSquared * resistance * resistance;

    return wattage;
}

int main() {
    double cadence = 0.0, resistance = 0.0;
    char input[100];

    printf("Enter cadence (%.1f - %.1f): ", MIN_CADENCE, MAX_CADENCE);
    if (fgets(input, sizeof(input), stdin) == NULL ||
        sscanf(input, "%lf", &cadence) != 1 ||
        cadence < MIN_CADENCE || cadence > MAX_CADENCE) {
        fprintf(stderr, "Invalid cadence input.\n");
        return 1;
    }

    printf("Enter resistance (%.1f - %.1f): ", MIN_RESISTANCE, MAX_RESISTANCE);
    if (fgets(input, sizeof(input), stdin) == NULL ||
        sscanf(input, "%lf", &resistance) != 1 ||
        resistance < MIN_RESISTANCE || resistance > MAX_RESISTANCE) {
        fprintf(stderr, "Invalid resistance input.\n");
        return 1;
    }

    double wattage = calculateWattage(cadence, resistance);

    if (!isfinite(wattage)) {
        fprintf(stderr, "Calculation error: wattage is not a finite number.\n");
        return 1;
    }

    printf("Predicted Wattage: %.2f Watts\n", wattage);
    return 0;
}