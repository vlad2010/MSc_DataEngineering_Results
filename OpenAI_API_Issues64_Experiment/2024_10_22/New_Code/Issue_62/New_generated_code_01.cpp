#include <iostream>

template <typename T>
class Stack
{
private:
    T *arr;
    unsigned int cnt, maxLen;
    T emptyValue;

public:
    Stack(unsigned int stackSize, const T &emptyVal);
    ~Stack();
    Stack(const Stack &other); // Copy constructor
    Stack &operator=(const Stack &other); // Assignment operator
    void push(const T &val);
    T pop();
};

template <typename T>
Stack<T>::Stack(unsigned int stackSize, const T &emptyVal)
{
    this->arr = new T[stackSize];
    this->cnt = 0;
    this->maxLen = stackSize;
    this->emptyValue = emptyVal;
}

template <typename T>
Stack<T>::~Stack()
{
    delete[] this->arr;
}

template <typename T>
Stack<T>::Stack(const Stack &other)
    : cnt(other.cnt), maxLen(other.maxLen), emptyValue(other.emptyValue)
{
    this->arr = new T[other.maxLen];
    std::copy(other.arr, other.arr + other.cnt, this->arr);
}

template <typename T>
Stack<T> &Stack<T>::operator=(const Stack &other)
{
    if (this != &other)
    {
        delete[] this->arr; // Free existing resource

        this->cnt = other.cnt;
        this->maxLen = other.maxLen;
        this->emptyValue = other.emptyValue;

        this->arr = new T[other.maxLen];
        std::copy(other.arr, other.arr + other.cnt, this->arr);
    }
    return *this;
}

template <typename T>
void Stack<T>::push(const T &val)
{
    if (this->cnt < this->maxLen)
        this->arr[this->cnt++] = val;
    else
        std::cerr << "Error: Stack overflow." << std::endl;
}

template <typename T>
T Stack<T>::pop()
{
    if (this->cnt)
        return this->arr[--this->cnt];
    else
        return this->emptyValue; // Ensure emptyValue is properly initialized
}