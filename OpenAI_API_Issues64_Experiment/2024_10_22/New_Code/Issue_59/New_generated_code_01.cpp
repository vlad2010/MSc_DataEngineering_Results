#ifndef _ALLIST_H_
#define _ALLIST_H_

#include <Arduino.h>
#include <memory>

template <typename T>
class Node {
public:
    std::shared_ptr<T> data; // Shared pointer to data
    std::unique_ptr<Node<T>> next; // Unique pointer to the next node

    Node(const std::shared_ptr<T> &data) : data(data), next(nullptr) {}
};

template <typename T>
class al_llist {
private:
    std::unique_ptr<Node<T>> head;
    mutable size_t frameMaxLen = 0;

    void mergeSort(std::unique_ptr<Node<T>>& headRef, int (*cmp)(T&, T&)) {
        if ((headRef == nullptr) || (headRef->next == nullptr)) {
            return;
        }

        std::unique_ptr<Node<T>> a;
        std::unique_ptr<Node<T>> b;

        frontBackSplit(headRef, a, b);
        mergeSort(a, cmp);
        mergeSort(b, cmp);
        headRef = merge(a, b, cmp);
    }

    std::unique_ptr<Node<T>> merge(std::unique_ptr<Node<T>>& a, std::unique_ptr<Node<T>>& b, int (*cmp)(T&, T&)) {
        if (a == nullptr) return std::move(b);
        else if (b == nullptr) return std::move(a);

        std::unique_ptr<Node<T>> result;
        if (cmp(*a->data, *b->data) <= 0) {
            result = std::move(a);
            result->next = merge(result->next, b, cmp);
        } else {
            result = std::move(b);
            result->next = merge(a, result->next, cmp);
        }
        return result;
    }

    void frontBackSplit(std::unique_ptr<Node<T>>& source, std::unique_ptr<Node<T>>& frontRef, std::unique_ptr<Node<T>>& backRef) {
        if (source == nullptr || source->next == nullptr) {
            frontRef = std::move(source);
            backRef = nullptr;
            return;
        }

        Node<T>* slow = source.get();
        Node<T>* fast = source->next.get();

        while (fast != nullptr) {
            fast = fast->next.get();
            if (fast != nullptr) {
                slow = slow->next.get();
                fast = fast->next.get();
            }
        }

        frontRef = std::move(source);
        backRef = std::move(slow->next);
    }

public:
    void sort(int (*cmp)(T&, T&)) {
        mergeSort(head, cmp);
    }
};

#endif // _ALLIST_H_