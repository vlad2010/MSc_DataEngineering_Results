#include <stdio.h>

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
    double cadence = 55;
    double resistance = 13;
    double wattage = calculateWattage(cadence, resistance);
    printf("Predicted Wattage: %.2f Watts\n", wattage);
    return 0;
}