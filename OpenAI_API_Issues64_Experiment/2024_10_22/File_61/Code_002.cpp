class BankAccount {
private:
    double balance;
    
    void deductFees() {
        // Some code to deduct fees from the balance
    }
    
public:
    void deposit(double amount) {
        balance += amount;
    }
    
    double getBalance() {
        return balance;
    }
};

int main() {
    BankAccount account;
    account.deposit(1000); // Accessing public method to deposit money
    // account.balance = 500; // Error: balance is private and not accessible
    // account.deductFees(); // Error: deductFees() is private and not accessible
    return 0;
}
