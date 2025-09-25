#include <iostream>
#include <mutex>
#include <limits>
#include <stdexcept>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <ctime>
#include <string>

class BankAccount {
private:
    // Use long long for cents to avoid floating-point precision issues
    long long balanceInCents;
    mutable std::mutex accountMutex;
    std::string accountId;
    static constexpr long long MAX_BALANCE_CENTS = 999999999999LL; // $9,999,999,999.99
    static constexpr long long MAX_TRANSACTION_CENTS = 100000000LL; // $1,000,000.00
    
    void deductFees(long long feesInCents) {
        std::lock_guard<std::mutex> lock(accountMutex);
        if (feesInCents < 0) {
            logTransaction("INVALID_FEE_DEDUCTION", 0, false, "Negative fee amount");
            throw std::invalid_argument("Fees cannot be negative");
        }
        if (feesInCents > balanceInCents) {
            logTransaction("INSUFFICIENT_FUNDS_FOR_FEES", feesInCents, false, "Insufficient balance");
            throw std::runtime_error("Insufficient balance for fees");
        }
        balanceInCents -= feesInCents;
        logTransaction("FEE_DEDUCTION", feesInCents, true, "");
    }
    
    void logTransaction(const std::string& type, long long amountCents, bool success, const std::string& reason) const {
        // In production, this would write to a secure audit log
        std::time_t now = std::time(nullptr);
        std::cout << "[AUDIT] " << std::ctime(&now);
        std::cout << "Account: " << accountId << ", Type: " << type 
                  << ", Amount: $" << (amountCents / 100.0) 
                  << ", Success: " << (success ? "YES" : "NO");
        if (!reason.empty()) {
            std::cout << ", Reason: " << reason;
        }
        std::cout << std::endl;
    }
    
    long long dollarsToCents(double dollars) const {
        // Round to nearest cent to handle floating-point precision
        return static_cast<long long>(std::round(dollars * 100));
    }
    
public:
    // Proper constructor with initialization
    explicit BankAccount(const std::string& id = "DEFAULT") 
        : balanceInCents(0), accountId(id) {
        logTransaction("ACCOUNT_CREATED", 0, true, "");
    }
    
    // Destructor for cleanup
    ~BankAccount() {
        logTransaction("ACCOUNT_CLOSED", balanceInCents, true, "");
    }
    
    // Delete copy operations to prevent accidental duplication
    BankAccount(const BankAccount&) = delete;
    BankAccount& operator=(const BankAccount&) = delete;
    
    void deposit(double amount) {
        std::lock_guard<std::mutex> lock(accountMutex);
        
        // Input validation
        if (!std::isfinite(amount)) {
            logTransaction("DEPOSIT", 0, false, "Invalid amount (NaN or Inf)");
            throw std::invalid_argument("Amount must be a valid number");
        }
        
        long long amountInCents = dollarsToCents(amount);
        
        if (amountInCents <= 0) {
            logTransaction("DEPOSIT", amountInCents, false, "Non-positive amount");
            throw std::invalid_argument("Deposit amount must be positive");
        }
        
        if (amountInCents > MAX_TRANSACTION_CENTS) {
            logTransaction("DEPOSIT", amountInCents, false, "Exceeds transaction limit");
            throw std::invalid_argument("Deposit exceeds maximum transaction limit");
        }
        
        // Check for overflow
        if (balanceInCents > MAX_BALANCE_CENTS - amountInCents) {
            logTransaction("DEPOSIT", amountInCents, false, "Would exceed maximum balance");
            throw std::overflow_error("Deposit would exceed maximum account balance");
        }
        
        balanceInCents += amountInCents;
        logTransaction("DEPOSIT", amountInCents, true, "");
    }
    
    void withdraw(double amount) {
        std::lock_guard<std::mutex> lock(accountMutex);
        
        if (!std::isfinite(amount)) {
            logTransaction("WITHDRAWAL", 0, false, "Invalid amount (NaN or Inf)");
            throw std::invalid_argument("Amount must be a valid number");
        }
        
        long long amountInCents = dollarsToCents(amount);
        
        if (amountInCents <= 0) {
            logTransaction("WITHDRAWAL", amountInCents, false, "Non-positive amount");
            throw std::invalid_argument("Withdrawal amount must be positive");
        }
        
        if (amountInCents > MAX_TRANSACTION_CENTS) {
            logTransaction("WITHDRAWAL", amountInCents, false, "Exceeds transaction limit");
            throw std::invalid_argument("Withdrawal exceeds maximum transaction limit");
        }
        
        if (amountInCents > balanceInCents) {
            logTransaction("WITHDRAWAL", amountInCents, false, "Insufficient funds");
            throw std::runtime_error("Insufficient funds");
        }
        
        balanceInCents -= amountInCents;
        logTransaction("WITHDRAWAL", amountInCents, true, "");
    }
    
    double getBalance() const {
        std::lock_guard<std::mutex> lock(accountMutex);
        return balanceInCents / 100.0;
    }
    
    std::string getBalanceAsString() const {
        std::lock_guard<std::mutex> lock(accountMutex);
        std::stringstream ss;
        ss << std::fixed << std::setprecision(2) << (balanceInCents / 100.0);
        return ss.str();
    }
};

int main() {
    try {
        BankAccount account("ACC123456");
        
        // Valid operations
        account.deposit(1000.00);
        std::cout << "Balance: $" << account.getBalanceAsString() << std::endl;
        
        account.withdraw(50.00);
        std::cout << "Balance after withdrawal: $" << account.getBalanceAsString() << std::endl;
        
        // These would throw exceptions (uncomment to test):
        // account.deposit(-100);  // Throws: negative amount
        // account.deposit(10000000);  // Throws: exceeds transaction limit
        // account.withdraw(2000);  // Throws: insufficient funds
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}