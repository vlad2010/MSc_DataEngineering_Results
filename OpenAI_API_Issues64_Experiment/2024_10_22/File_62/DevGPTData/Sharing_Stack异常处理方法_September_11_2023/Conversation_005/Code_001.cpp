template <typename T>
class Stack
{
private:
    T *arr;
    unsigned int cnt, maxLen;
    T emptyValue; // 自定义类T的表示空值的实例

public:
    Stack(unsigned int stackSize, const T &emptyVal);
    ~Stack();
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
void Stack<T>::push(const T &val)
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
        return this->emptyValue; // 返回自定义类T的表示空值的实例
}
