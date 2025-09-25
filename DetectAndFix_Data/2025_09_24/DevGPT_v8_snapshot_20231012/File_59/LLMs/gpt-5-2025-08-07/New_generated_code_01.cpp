#ifndef _ALLIST_H_
#define _ALLIST_H_

#include <Arduino.h>
#include <memory>
#include <cstddef>

template <typename T>
class Node
{
public:
  std::shared_ptr<T> data;            // Shared pointer to payload
  std::unique_ptr<Node<T>> next;      // Ownership of next node

  explicit Node(const std::shared_ptr<T> &data_) : data(data_), next(nullptr) {}
};

template <typename T>
class al_llist
{
private:
  std::unique_ptr<Node<T>> head;
  mutable size_t frameMaxLen = 0;

  // Safe comparator wrapper: handles null data pointers
  static int safeCmp(const std::shared_ptr<T>& lhs,
                     const std::shared_ptr<T>& rhs,
                     int (*cmp)(const T&, const T&))
  {
    if (!lhs && !rhs) return 0;
    if (!lhs) return -1;   // nulls come before non-nulls
    if (!rhs) return 1;
    return cmp(*lhs, *rhs);
  }

  // Return the length of the list (no ownership changes).
  size_t length() const
  {
    size_t len = 0;
    for (auto p = head.get(); p; p = p->next.get()) ++len;
    return len;
  }

  // Split off the first 'n' nodes from 'src' (if available), returning the remainder.
  // Ownership: 'src' remains the first segment of up to n nodes, returned value is the remainder.
  static std::unique_ptr<Node<T>> splitN(std::unique_ptr<Node<T>>& src, size_t n)
  {
    if (!src) return nullptr;
    Node<T>* curr = src.get();
    while (n > 1 && curr->next) {
      curr = curr->next.get();
      --n;
    }
    std::unique_ptr<Node<T>> rest = std::move(curr->next); // detach tail
    // curr->next is now nullptr
    return rest;
  }

  struct MergeResult {
    std::unique_ptr<Node<T>> head;
    Node<T>* tail; // raw, non-owning pointer to last node for efficient appends
  };

  // Iteratively merge two sorted lists 'a' and 'b' using unique_ptr moves (no recursion).
  static MergeResult mergeLists(std::unique_ptr<Node<T>> a,
                                std::unique_ptr<Node<T>> b,
                                int (*cmp)(const T&, const T&))
  {
    std::unique_ptr<Node<T>> result;
    Node<T>* tail = nullptr;

    auto take = [&](std::unique_ptr<Node<T>>& src) {
      std::unique_ptr<Node<T>> next = std::move(src->next);
      if (!result) {
        result = std::move(src);
        tail = result.get();
      } else {
        tail->next = std::move(src);
        tail = tail->next.get();
      }
      src = std::move(next);
    };

    while (a && b) {
      if (safeCmp(a->data, b->data, cmp) <= 0) {
        take(a);
      } else {
        take(b);
      }
    }

    // Append the remainder
    if (a) {
      if (!result) { result = std::move(a); tail = result.get(); }
      else { tail->next = std::move(a); }
    } else if (b) {
      if (!result) { result = std::move(b); tail = result.get(); }
      else { tail->next = std::move(b); }
    }

    // Advance tail to the end of the appended list (only needed once per merge)
    if (tail) {
      while (tail->next) tail = tail->next.get();
    }

    return { std::move(result), tail };
  }

public:
  al_llist() = default;

  // Example insert at front (safe utility; not required but useful for testing)
  void push_front(const std::shared_ptr<T>& value)
  {
    auto n = std::make_unique<Node<T>>(value);
    n->next = std::move(head);
    head = std::move(n);
  }

  // Public sort: iterative bottom-up mergesort to avoid recursion (CWE-674).
  // Comparator must not be null and should be a strict weak ordering.
  void sort(int (*cmp)(const T&, const T&))
  {
    if (!cmp) {
      // Defensive: nothing to do if comparator is null (CWE-476)
      return;
    }
    if (!head || !head->next) {
      return; // 0 or 1 element
    }

    size_t n = length();
    for (size_t width = 1; width < n; width <<= 1) {
      std::unique_ptr<Node<T>> newHead;
      Node<T>* newTail = nullptr;

      std::unique_ptr<Node<T>> remaining = std::move(head);
      while (remaining) {
        // Cut two runs of size 'width': left and right
        std::unique_ptr<Node<T>> left = std::move(remaining);
        std::unique_ptr<Node<T>> right = splitN(left, width);
        remaining = splitN(right, width);

        // Merge left and right, append to new list
        MergeResult merged = mergeLists(std::move(left), std::move(right), cmp);
        if (!newHead) {
          newHead = std::move(merged.head);
          newTail = merged.tail;
        } else {
          newTail->next = std::move(merged.head);
          if (newTail->next) newTail = merged.tail;
        }
      }
      head = std::move(newHead);
    }
  }

  // Optional: expose head for testing/iteration (const)
  const Node<T>* cbegin() const { return head.get(); }
};

#endif // _ALLIST_H_