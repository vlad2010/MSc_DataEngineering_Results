#include <iostream>
#include <functional>

int main() {
    // Define a std::function object for the recursive lambda
    std::function<int(int)> factorial;

    // Assign a lambda to the std::function object
    factorial = [&factorial](int n) -> int {
        if (n <= 1) return 1;
        else return n * factorial(n - 1);
    };

    // Use the lambda function
    std::cout << "Factorial of 5 is " << factorial(5) << std::endl;

    return 0;
}