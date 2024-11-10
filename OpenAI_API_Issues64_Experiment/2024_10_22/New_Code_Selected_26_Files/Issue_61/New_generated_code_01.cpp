#include <iostream>
#include <mutex>
#include <stdexcept>

class BankAccount {
private:
    double balance;
    std::mutex mtx;  // Mutex to ensure thread safety
    
    void deductFees() {
        // Assume some logic is here to deduct fees
        // e.g., balance -= feeAmount;
        // This method should handle cases safely and possibly be called under certain conditions.
        std::lock_guard<std::mutex> lock(mtx);
        double feeAmount = 10;  // Example fee deduction
        if (balance >= feeAmount) {
            balance -= feeAmount;
        } else {
            throw std::runtime_error("Insufficient funds to deduct fee.");
        }
    }

public:
    BankAccount() : balance(0.0) {}

    void deposit(double amount) {
        if (amount <= 0) {  // Input validation to prevent invalid deposits
            throw std::invalid_argument("Deposit amount must be positive.");
        }
        std::lock_guard<std::mutex> lock(mtx);
        balance += amount;
    }
    
    double getBalance() {
        std::lock_guard<std::mutex> lock(mtx);
        return balance;
    }

    // Public method to allow users to call for fee deduction.
    // This could also be scheduled/automated through other class logic
    void applyFees() {
        deductFees();
    }
};

int main() {
    BankAccount account;
    try {
        account.deposit(1000);  // Accessing public method to deposit money
        account.applyFees();    // Now you can deduct fees through a controlled method
        std::cout << "Current Balance: " << account.getBalance() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return 0;
}