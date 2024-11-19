#include "allist.h"

// Fonction de comparaison pour trier la liste d'entiers en ordre croissant
int compareInt(const int &a, const int &b) {
    if (a < b) return -1;
    if (a > b) return 1;
    return 0;
}

void setup() {
    Serial.begin(9600);

    // Création d'une instance de al_llist pour stocker des entiers
    al_llist<int> myList;

    // Ajout d'entiers à la liste
    // (Supposons que vous ayez une méthode 'append' pour ajouter des éléments à la liste)
    myList.append(5);
    myList.append(3);
    myList.append(8);
    myList.append(1);
    myList.append(4);

    // Affichage de la liste non triée
    Serial.println("Liste non triée:");
    myList.display();

    // Tri de la liste en utilisant la méthode sort
    myList.sort(compareInt);

    // Affichage de la liste triée
    Serial.println("Liste triée:");
    myList.display();
}

void loop() {
    // Votre code de boucle ici
}
