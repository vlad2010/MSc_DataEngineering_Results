#ifndef _ALLIST_H_
#define _ALLIST_H_

#include <Arduino.h>
#include <memory>
#include <functional>

template <typename T>
class Node
{
public:
  std::shared_ptr<T> data;
  std::unique_ptr<Node<T>> next;

  Node(const std::shared_ptr<T> &data) : data(data), next(nullptr) {}
};

template <typename T>
class al_llist
{
private:
  std::unique_ptr<Node<T>> head;
  mutable size_t frameMaxLen = 0;

  // Merge two sorted lists, transferring ownership
  std::unique_ptr<Node<T>> merge(std::unique_ptr<Node<T>> a, std::unique_ptr<Node<T>> b, const std::function<int(const T&, const T&)>& cmp)
  {
    if (!a) return b;
    if (!b) return a;

    if (cmp(*a->data, *b->data) <= 0)
    {
      a->next = merge(std::move(a->next), std::move(b), cmp);
      return a;
    }
    else
    {
      b->next = merge(std::move(a), std::move(b->next), cmp);
      return b;
    }
  }

  // Split the list into two halves, transferring ownership
  void frontBackSplit(std::unique_ptr<Node<T>>& source, std::unique_ptr<Node<T>>& frontRef, std::unique_ptr<Node<T>>& backRef)
  {
    if (!source || !source->next)
    {
      frontRef = std::move(source);
      backRef = nullptr;
      return;
    }

    Node<T>* slow = source.get();
    Node<T>* fast = source->next.get();

    while (fast && fast->next)
    {
      slow = slow->next.get();
      fast = fast->next->next.get();
    }

    // Split the list
    frontRef = std::move(source);
    backRef = std::move(slow->next);
    slow->next = nullptr;
  }

  void mergeSort(std::unique_ptr<Node<T>>& headRef, const std::function<int(const T&, const T&)>& cmp)
  {
    if (!headRef || !headRef->next)
      return;

    std::unique_ptr<Node<T>> a;
    std::unique_ptr<Node<T>> b;

    frontBackSplit(headRef, a, b);

    mergeSort(a, cmp);
    mergeSort(b, cmp);

    headRef = merge(std::move(a), std::move(b), cmp);
  }

public:
  // ... [Other public methods]

  void sort(const std::function<int(const T&, const T&)>& cmp)
  {
    mergeSort(head, cmp);
  }
};

#endif // _ALLIST_H_