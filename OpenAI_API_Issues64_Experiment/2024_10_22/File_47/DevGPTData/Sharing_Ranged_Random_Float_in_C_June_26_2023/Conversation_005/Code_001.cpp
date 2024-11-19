#include <iostream>
#include <list>

int main() {
    std::list<int> myList = {1, 2, 3, 4};

    // Using a range-based for loop
    for (int num : myList) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    // Using an iterator
    std::list<int>::iterator it;
    for (it = myList.begin(); it != myList.end(); ++it) {
        std::cout << *it << " ";
    }
    std::cout << std::endl;

    return 0;
}
