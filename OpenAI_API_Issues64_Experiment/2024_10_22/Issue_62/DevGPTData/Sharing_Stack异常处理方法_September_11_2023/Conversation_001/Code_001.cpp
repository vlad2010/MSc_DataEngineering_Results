#include <iostream>

// 自定义栈为空的异常类
class StackEmptyException : public std::exception
{
public:
    const char *what() const throw()
    {
        return "Stack is empty";
    }
};

template <typename T>
class Stack
{
private:
    T *arr;
    unsigned int cnt, maxLen;

public:
    Stack(unsigned int stackSize);
    ~Stack();
    void push(T val);
    T pop();
};

template <typename T>
Stack<T>::Stack(unsigned int stackSize)
{
    this->arr = new T[stackSize];
    this->cnt = 0;
    this->maxLen = stackSize;
}

template <typename T>
Stack<T>::~Stack()
{
    delete[] this->arr;
}

template <typename T>
void Stack<T>::push(T val)
{
    if (this->cnt < this->maxLen)
        this->arr[this->cnt++] = val;
}

template <typename T>
T Stack<T>::pop()
{
    if (this->cnt)
        return this->arr[--this->cnt];
    else
        throw StackEmptyException(); // 抛出栈为空的异常
}

int main()
{
    Stack<int> stack(5);

    try
    {
        stack.pop(); // 尝试从空栈中弹出元素
    }
    catch (const std::exception &ex)
    {
        std::cerr << "Exception: " << ex.what() << std::endl;
    }

    return 0;
}
