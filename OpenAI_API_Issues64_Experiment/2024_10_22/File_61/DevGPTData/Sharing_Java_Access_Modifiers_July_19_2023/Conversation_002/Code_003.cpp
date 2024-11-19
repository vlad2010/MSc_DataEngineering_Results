class Animal {
protected:
    std::string name;
    
    void makeSound() {
        std::cout << "Animal makes a sound." << std::endl;
    }
};

class Dog : public Animal {
public:
    void greet() {
        std::cout << "Dog barks, says hello!" << std::endl;
        makeSound(); // Accessing protected method from the base class
    }
};

int main() {
    Dog dog;
    dog.name = "Buddy"; // Accessing protected variable from the subclass
    dog.greet(); // Accessing public method that indirectly calls the protected method
    // dog.makeSound(); // Error: makeSound() is protected and not accessible from outside
    return 0;
}
