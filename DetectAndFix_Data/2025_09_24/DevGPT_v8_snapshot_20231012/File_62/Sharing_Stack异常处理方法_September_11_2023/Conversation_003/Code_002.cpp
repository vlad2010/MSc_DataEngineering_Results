template <typename T>
bool Stack<T>::pop(T &result)
{
    if (this->cnt)
    {
        result = this->arr[--this->cnt];
        return true;
    }
    else
    {
        return false; // 操作失败，栈为空
    }
}
