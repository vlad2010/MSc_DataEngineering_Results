// ... [Autres parties de la classe al_llist]

private:

  void mergeSort(std::unique_ptr<Node<T>>& headRef, int (*cmp)(T&, T&))
  {
    if (!headRef || !headRef->next) {
        return;
    }

    std::unique_ptr<Node<T>> a;
    std::unique_ptr<Node<T>> b;

    frontBackSplit(headRef, a, b);

    mergeSort(a, cmp);
    mergeSort(b, cmp);

    headRef = merge(std::move(a), std::move(b), cmp);
  }

  std::unique_ptr<Node<T>> merge(std::unique_ptr<Node<T>> a, std::unique_ptr<Node<T>> b, int (*cmp)(T&, T&))
  {
    if (!a) return b;
    if (!b) return a;

    std::unique_ptr<Node<T>> result;

    if (cmp(*a->data, *b->data) <= 0)
    {
        result = std::move(a);
        result->next = merge(std::move(result->next), std::move(b), cmp);
    }
    else
    {
        result = std::move(b);
        result->next = merge(std::move(a), std::move(result->next), cmp);
    }

    return result;
  }

  void frontBackSplit(std::unique_ptr<Node<T>>& source, std::unique_ptr<Node<T>>& frontRef, std::unique_ptr<Node<T>>& backRef)
  {
    Node<T>* fast = source->next.get();
    Node<T>* slow = source.get();

    while (fast)
    {
        fast = fast->next ? fast->next->next : nullptr;
        if (fast) slow = slow->next.get();
    }

    frontRef = std::move(source);
    backRef = std::move(slow->next);
  }

public:

  void sort(int (*cmp)(T&, T&))
  {
    mergeSort(head, cmp);
  }
