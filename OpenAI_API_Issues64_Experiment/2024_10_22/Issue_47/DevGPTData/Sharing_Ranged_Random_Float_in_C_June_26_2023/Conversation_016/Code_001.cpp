#include <iostream>
#include <string>

class MyClass {
public:
    int myInt;
    std::string myString;

    // Constructor
    MyClass(int value, const std::string& str) : myInt(value), myString(str) {}
};

int main() {
    // Creating objects on the stack
    MyClass obj1(10, "Hello");
    MyClass obj2(20, "World");

    // Accessing object members
    std::cout << obj1.myInt << " " << obj1.myString << std::endl;
    std::cout << obj2.myInt << " " << obj2.myString << std::endl;

    return 0;
}
