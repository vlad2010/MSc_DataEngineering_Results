#include <iostream>
#include <limits>
#include <stdexcept>
#include <cmath>
#include <mutex>

class BankAccount {
private:
    double balance;
    mutable std::mutex mtx; // For thread safety

    void deductFees() {
        // Some code to deduct fees from the balance
        // Example: balance -= 10.0;
    }

public:
    // Constructor initializes balance to zero
    BankAccount() : balance(0.0) {}

    // Deposit method with input validation and overflow checks
    void deposit(double amount) {
        std::lock_guard<std::mutex> lock(mtx); // Thread safety

        // Input validation: amount must be positive and finite
        if (amount <= 0.0 || !std::isfinite(amount)) {
            throw std::invalid_argument("Deposit amount must be positive and finite.");
        }

        // Overflow check
        if (balance > std::numeric_limits<double>::max() - amount) {
            throw std::overflow_error("Deposit would cause balance overflow.");
        }

        balance += amount;
    }

    // Getter for balance (thread-safe)
    double getBalance() const {
        std::lock_guard<std::mutex> lock(mtx);
        return balance;
    }
};

int main() {
    try {
        BankAccount account;
        account.deposit(1000); // Accessing public method to deposit money
        // account.balance = 500; // Error: balance is private and not accessible
        // account.deductFees(); // Error: deductFees() is private and not accessible
        std::cout << "Balance: " << account.getBalance() << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
    }
    return 0;
}