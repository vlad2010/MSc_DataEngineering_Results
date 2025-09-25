#include <stdexcept>
#include <algorithm> // for std::swap

template <typename T>
class Stack
{
private:
    T *arr;
    unsigned int cnt, maxLen;
    T emptyValue;

    void copyFrom(const Stack &other) {
        arr = new T[other.maxLen];
        for (unsigned int i = 0; i < other.cnt; ++i)
            arr[i] = other.arr[i];
        cnt = other.cnt;
        maxLen = other.maxLen;
        emptyValue = other.emptyValue;
    }

public:
    // Constructor
    Stack(unsigned int stackSize, const T &emptyVal)
        : arr(nullptr), cnt(0), maxLen(stackSize), emptyValue(emptyVal)
    {
        if (stackSize == 0)
            throw std::invalid_argument("Stack size must be greater than zero");
        arr = new T[stackSize];
    }

    // Destructor
    ~Stack()
    {
        delete[] arr;
    }

    // Copy constructor
    Stack(const Stack &other)
    {
        copyFrom(other);
    }

    // Copy assignment
    Stack& operator=(const Stack &other)
    {
        if (this != &other) {
            delete[] arr;
            copyFrom(other);
        }
        return *this;
    }

    // Move constructor
    Stack(Stack &&other) noexcept
        : arr(other.arr), cnt(other.cnt), maxLen(other.maxLen), emptyValue(std::move(other.emptyValue))
    {
        other.arr = nullptr;
        other.cnt = 0;
        other.maxLen = 0;
    }

    // Move assignment
    Stack& operator=(Stack &&other) noexcept
    {
        if (this != &other) {
            delete[] arr;
            arr = other.arr;
            cnt = other.cnt;
            maxLen = other.maxLen;
            emptyValue = std::move(other.emptyValue);
            other.arr = nullptr;
            other.cnt = 0;
            other.maxLen = 0;
        }
        return *this;
    }

    // Push returns bool to indicate success/failure
    bool push(const T &val)
    {
        if (cnt < maxLen) {
            arr[cnt++] = val;
            return true;
        }
        return false; // Stack overflow
    }

    // Pop throws exception on underflow
    T pop()
    {
        if (cnt == 0)
            throw std::underflow_error("Stack underflow");
        return arr[--cnt];
    }

    // Optional: peek method
    T peek() const
    {
        if (cnt == 0)
            throw std::underflow_error("Stack is empty");
        return arr[cnt - 1];
    }

    // Optional: size method
    unsigned int size() const { return cnt; }
    unsigned int capacity() const { return maxLen; }
    bool empty() const { return cnt == 0; }
    bool full() const { return cnt == maxLen; }
};