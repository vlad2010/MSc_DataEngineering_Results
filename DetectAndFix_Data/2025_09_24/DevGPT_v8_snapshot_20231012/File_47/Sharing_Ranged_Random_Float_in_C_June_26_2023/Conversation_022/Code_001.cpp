#include <iostream>

class MyClass {
public:
    MyClass(int value) : data(value) {}

    void printData() const {
        std::cout << "Data: " << data << std::endl;
    }

    void assignObject(const MyClass& other) {
        data = other.data; // Copy assignment
    }

    void assignObject(MyClass&& other) {
        data = std::move(other.data); // Move assignment
    }

private:
    int data;
};

int main() {
    MyClass obj1(42); // Create an object on the stack

    MyClass obj2(0);
    obj2.assignObject(obj1); // Assign obj1 to obj2 using copy assignment
    obj2.printData(); // Output: Data: 42

    MyClass obj3(0);
    obj3.assignObject(std::move(obj1)); // Assign obj1 to obj3 using move assignment
    obj3.printData(); // Output: Data: 42

    return 0;
}
