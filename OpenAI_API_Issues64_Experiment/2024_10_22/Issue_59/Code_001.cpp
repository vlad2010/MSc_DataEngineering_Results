#ifndef _ALLIST_H_
#define _ALLIST_H_

#include <Arduino.h>
#include <memory>

template <typename T>
class Node
{
public:
  std::shared_ptr<T> data; // Pointeur partagé vers les données
  std::unique_ptr<Node<T>> next; // Pointeur unique vers le nœud suivant

  Node(const std::shared_ptr<T> &data) : data(data), next(nullptr) {}
};

template <typename T>
class al_llist
{
private:
  std::unique_ptr<Node<T>> head; 
  mutable size_t frameMaxLen = 0;

  // ... [Les autres méthodes et membres de la classe]

  // Ajout des méthodes pour le tri par fusion
  void mergeSort(std::unique_ptr<Node<T>>& headRef, int (*cmp)(T&, T&))
  {
    Node<T>* head = headRef.get();
    Node<T>* a;
    Node<T>* b;

    if ((head == nullptr) || (head->next == nullptr))
    {
        return;
    }

    frontBackSplit(head, &a, &b);

    mergeSort(a, cmp);
    mergeSort(b, cmp);

    headRef.reset(merge(a, b, cmp));
  }

  Node<T>* merge(Node<T>* a, Node<T>* b, int (*cmp)(T&, T&))
  {
    Node<T>* result = nullptr;

    if (a == nullptr)
        return b;
    else if (b == nullptr)
        return a;

    if (cmp(*a->data, *b->data) <= 0)
    {
        result = a;
        result->next.reset(merge(a->next.get(), b, cmp));
    }
    else
    {
        result = b;
        result->next.reset(merge(a, b->next.get(), cmp));
    }
    return result;
  }

  void frontBackSplit(Node<T>* source, Node<T>** frontRef, Node<T>** backRef)
  {
    Node<T>* fast;
    Node<T>* slow;
    slow = source;
    fast = source->next.get();

    while (fast != nullptr)
    {
        fast = fast->next.get();
        if (fast != nullptr)
        {
            slow = slow->next.get();
            fast = fast->next.get();
        }
    }

    *frontRef = source;
    *backRef = slow->next.get();
    slow->next = nullptr;
  }

public:
  // ... [Les autres méthodes publiques de la classe]

  void sort(int (*cmp)(T&, T&))
  {
    mergeSort(head, cmp);
  }
};

#endif // _ALLIST_H_
