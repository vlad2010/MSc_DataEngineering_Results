class Car {
public:
    std::string brand;
    
    void start() {
        std::cout << "Car started." << std::endl;
    }
};

int main() {
    Car myCar;
    myCar.brand = "Toyota"; // Accessing public variable
    myCar.start(); // Accessing public method
    return 0;
}
