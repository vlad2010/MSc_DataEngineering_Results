template <typename T>
T Stack<T>::pop()
{
    if (this->cnt)
        return this->arr[--this->cnt];
    else
        return /* 返回特定的标志值，例如 -1 */ -1;
}
