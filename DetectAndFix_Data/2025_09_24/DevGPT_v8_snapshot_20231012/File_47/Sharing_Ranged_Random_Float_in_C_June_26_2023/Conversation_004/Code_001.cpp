#include <iostream>
#include <list>

int main() {
    std::list<int> myList = {1, 2, 3, 4};

    // Insert an element at the beginning of the list
    myList.insert(myList.begin(), 0);

    // Insert an element at a specific position
    std::list<int>::iterator it = std::next(myList.begin(), 2);
    myList.insert(it, 5);

    // Insert multiple elements at a specific position
    std::list<int> newElements = {6, 7, 8};
    myList.insert(it, newElements.begin(), newElements.end());

    // Print the elements of the list
    for (int num : myList) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    return 0;
}
