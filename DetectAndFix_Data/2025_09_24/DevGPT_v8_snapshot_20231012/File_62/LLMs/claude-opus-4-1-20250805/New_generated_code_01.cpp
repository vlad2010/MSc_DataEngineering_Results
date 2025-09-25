#include <algorithm>
#include <stdexcept>
#include <limits>
#include <new>

template <typename T>
class Stack
{
private:
    T *arr;
    unsigned int cnt, maxLen;
    T emptyValue;
    
    // Constants for validation
    static constexpr unsigned int MIN_STACK_SIZE = 1;
    static constexpr unsigned int MAX_STACK_SIZE = 1000000; // Reasonable upper limit

public:
    // Constructor
    Stack(unsigned int stackSize, const T &emptyVal);
    
    // Destructor
    ~Stack();
    
    // Copy constructor (Rule of Five)
    Stack(const Stack<T>& other);
    
    // Copy assignment operator (Rule of Five)
    Stack<T>& operator=(const Stack<T>& other);
    
    // Move constructor (Rule of Five)
    Stack(Stack<T>&& other) noexcept;
    
    // Move assignment operator (Rule of Five)
    Stack<T>& operator=(Stack<T>&& other) noexcept;
    
    // Stack operations
    void push(const T &val);
    T pop();
    bool isEmpty() const { return cnt == 0; }
    bool isFull() const { return cnt >= maxLen; }
    unsigned int size() const { return cnt; }
};

template <typename T>
Stack<T>::Stack(unsigned int stackSize, const T &emptyVal)
    : arr(nullptr), cnt(0), maxLen(0), emptyValue(emptyVal)
{
    // Validate stack size to prevent integer overflow and unreasonable allocations
    if (stackSize < MIN_STACK_SIZE || stackSize > MAX_STACK_SIZE) {
        throw std::invalid_argument("Stack size must be between 1 and 1000000");
    }
    
    // Check for potential overflow in allocation size
    if (stackSize > std::numeric_limits<size_t>::max() / sizeof(T)) {
        throw std::overflow_error("Stack size too large for allocation");
    }
    
    try {
        this->arr = new T[stackSize];
        this->maxLen = stackSize;
    } catch (const std::bad_alloc& e) {
        // Clean up and rethrow
        this->arr = nullptr;
        this->maxLen = 0;
        throw std::runtime_error("Failed to allocate memory for stack");
    }
}

template <typename T>
Stack<T>::~Stack()
{
    delete[] this->arr;
    this->arr = nullptr;
}

// Copy constructor
template <typename T>
Stack<T>::Stack(const Stack<T>& other)
    : arr(nullptr), cnt(other.cnt), maxLen(other.maxLen), emptyValue(other.emptyValue)
{
    if (other.maxLen > 0 && other.arr != nullptr) {
        try {
            this->arr = new T[other.maxLen];
            // Copy elements
            for (unsigned int i = 0; i < other.cnt; ++i) {
                this->arr[i] = other.arr[i];
            }
        } catch (const std::bad_alloc& e) {
            this->arr = nullptr;
            this->cnt = 0;
            this->maxLen = 0;
            throw std::runtime_error("Failed to allocate memory in copy constructor");
        }
    }
}

// Copy assignment operator
template <typename T>
Stack<T>& Stack<T>::operator=(const Stack<T>& other)
{
    if (this != &other) {
        // Create temporary copy
        T* newArr = nullptr;
        
        if (other.maxLen > 0 && other.arr != nullptr) {
            try {
                newArr = new T[other.maxLen];
                // Copy elements
                for (unsigned int i = 0; i < other.cnt; ++i) {
                    newArr[i] = other.arr[i];
                }
            } catch (const std::bad_alloc& e) {
                delete[] newArr;
                throw std::runtime_error("Failed to allocate memory in copy assignment");
            }
        }
        
        // Delete old array
        delete[] this->arr;
        
        // Assign new values
        this->arr = newArr;
        this->cnt = other.cnt;
        this->maxLen = other.maxLen;
        this->emptyValue = other.emptyValue;
    }
    return *this;
}

// Move constructor
template <typename T>
Stack<T>::Stack(Stack<T>&& other) noexcept
    : arr(other.arr), cnt(other.cnt), maxLen(other.maxLen), emptyValue(std::move(other.emptyValue))
{
    other.arr = nullptr;
    other.cnt = 0;
    other.maxLen = 0;
}

// Move assignment operator
template <typename T>
Stack<T>& Stack<T>::operator=(Stack<T>&& other) noexcept
{
    if (this != &other) {
        // Delete current array
        delete[] this->arr;
        
        // Move resources
        this->arr = other.arr;
        this->cnt = other.cnt;
        this->maxLen = other.maxLen;
        this->emptyValue = std::move(other.emptyValue);
        
        // Reset other
        other.arr = nullptr;
        other.cnt = 0;
        other.maxLen = 0;
    }
    return *this;
}

template <typename T>
void Stack<T>::push(const T &val)
{
    if (this->arr == nullptr) {
        throw std::runtime_error("Stack is not properly initialized");
    }
    
    if (this->cnt < this->maxLen) {
        this->arr[this->cnt++] = val;
    } else {
        throw std::overflow_error("Stack overflow: cannot push to full stack");
    }
}

template <typename T>
T Stack<T>::pop()
{
    if (this->arr == nullptr) {
        throw std::runtime_error("Stack is not properly initialized");
    }
    
    if (this->cnt > 0) {
        return this->arr[--this->cnt];
    } else {
        return this->emptyValue;
    }
}