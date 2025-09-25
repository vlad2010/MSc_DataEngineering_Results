#ifndef _ALLIST_H_
#define _ALLIST_H_

#include <Arduino.h>
#include <memory>
#include <vector>
#include <algorithm>

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
    static constexpr size_t MAX_RECURSION_DEPTH = 1000; // Prevent stack overflow

    // Iterative merge sort to avoid stack overflow (CWE-674)
    std::unique_ptr<Node<T>> mergeSort(std::unique_ptr<Node<T>> headRef, int (*cmp)(T&, T&))
    {
        if (!headRef || !headRef->next) {
            return headRef;
        }

        // Convert to vector for safe sorting
        std::vector<std::pair<std::shared_ptr<T>, std::unique_ptr<Node<T>>>> nodes;
        
        // Extract all nodes safely
        while (headRef) {
            nodes.emplace_back(headRef->data, std::move(headRef->next));
            headRef = std::move(nodes.back().second);
            nodes.back().second = nullptr;
        }

        // Sort using stable_sort to maintain relative order of equal elements
        std::stable_sort(nodes.begin(), nodes.end(), 
            [cmp](const auto& a, const auto& b) {
                if (!a.first || !b.first) return false; // Null check (CWE-476)
                return cmp(*a.first, *b.first) < 0;
            });

        // Rebuild the list
        std::unique_ptr<Node<T>> newHead = nullptr;
        Node<T>* tail = nullptr;

        for (auto& [data, _] : nodes) {
            auto newNode = std::make_unique<Node<T>>(data);
            if (!newHead) {
                newHead = std::move(newNode);
                tail = newHead.get();
            } else {
                tail->next = std::move(newNode);
                tail = tail->next.get();
            }
        }

        return newHead;
    }

    // Alternative: Iterative bottom-up merge sort if vector approach is not desired
    std::unique_ptr<Node<T>> mergeSortBottomUp(std::unique_ptr<Node<T>> headRef, int (*cmp)(T&, T&))
    {
        if (!headRef || !headRef->next) {
            return headRef;
        }

        // Count nodes safely with iteration limit
        size_t count = 0;
        Node<T>* current = headRef.get();
        const size_t MAX_NODES = 100000; // Prevent infinite loops
        
        while (current && count < MAX_NODES) {
            count++;
            current = current->next.get();
        }

        if (count >= MAX_NODES) {
            // Log error or throw exception for list too large
            return headRef; // Return unsorted for safety
        }

        // Bottom-up iterative merge sort
        for (size_t size = 1; size < count; size *= 2) {
            std::unique_ptr<Node<T>> dummy = std::make_unique<Node<T>>(nullptr);
            Node<T>* tail = dummy.get();
            Node<T>* start = headRef.release();

            while (start) {
                Node<T>* mid = start;
                Node<T>* end = start;
                
                // Find mid and end points
                for (size_t i = 0; i < size && mid; i++) {
                    mid = mid->next.release();
                }
                
                if (!mid) {
                    tail->next.reset(start);
                    break;
                }
                
                for (size_t i = 0; i < size && end; i++) {
                    end = end->next.release();
                }

                // Merge two sorted segments
                tail->next = mergeIterative(start, mid, end, size, size, cmp);
                
                // Move tail to end of merged segment
                while (tail->next) {
                    tail = tail->next.get();
                }
                
                start = end;
            }
            
            headRef = std::move(dummy->next);
        }

        return headRef;
    }

    std::unique_ptr<Node<T>> mergeIterative(Node<T>* left, Node<T>* right, Node<T>* end,
                                           size_t leftSize, size_t rightSize, int (*cmp)(T&, T&))
    {
        std::unique_ptr<Node<T>> result = nullptr;
        Node<T>* tail = nullptr;
        
        while (leftSize > 0 && rightSize > 0 && left && right) {
            Node<T>* next = nullptr;
            
            // Null check for data (CWE-476)
            if (!left->data || !right->data) {
                break;
            }
            
            if (cmp(*left->data, *right->data) <= 0) {
                next = left;
                left = left->next.release();
                leftSize--;
            } else {
                next = right;
                right = right->next.release();
                rightSize--;
            }
            
            next->next = nullptr;
            
            if (!result) {
                result.reset(next);
                tail = result.get();
            } else {
                tail->next.reset(next);
                tail = tail->next.get();
            }
        }
        
        // Append remaining elements
        while (leftSize > 0 && left) {
            Node<T>* next = left;
            left = left->next.release();
            next->next = nullptr;
            
            if (!result) {
                result.reset(next);
                tail = result.get();
            } else {
                tail->next.reset(next);
                tail = tail->next.get();
            }
            leftSize--;
        }
        
        while (rightSize > 0 && right && right != end) {
            Node<T>* next = right;
            right = right->next.release();
            next->next = nullptr;
            
            if (!result) {
                result.reset(next);
                tail = result.get();
            } else {
                tail->next.reset(next);
                tail = tail->next.get();
            }
            rightSize--;
        }
        
        // Reconnect to the rest of the list
        if (tail && end) {
            tail->next.reset(end);
        }
        
        return result;
    }

public:
    al_llist() : head(nullptr), frameMaxLen(0) {}
    
    ~al_llist() {
        // Iterative destruction to avoid stack overflow on large lists
        while (head) {
            head = std::move(head->next);
        }
    }

    void sort(int (*cmp)(T&, T&))
    {
        if (!cmp) {
            return; // Null function pointer check (CWE-476)
        }
        
        // Use the safe vector-based approach
        head = mergeSort(std::move(head), cmp);
    }
    
    // Add safe insertion method
    void push_front(const std::shared_ptr<T>& data)
    {
        if (!data) {
            return; // Null check
        }
        
        auto newNode = std::make_unique<Node<T>>(data);
        newNode->next = std::move(head);
        head = std::move(newNode);
    }
    
    // Add safe size calculation with limit
    size_t size() const
    {
        size_t count = 0;
        const size_t MAX_COUNT = 1000000;
        Node<T>* current = head.get();
        
        while (current && count < MAX_COUNT) {
            count++;
            current = current->next.get();
        }
        
        return count;
    }
};

#endif // _ALLIST_H_