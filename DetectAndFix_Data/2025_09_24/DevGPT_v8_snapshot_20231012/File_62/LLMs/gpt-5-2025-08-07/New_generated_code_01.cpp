#include <vector>
#include <stdexcept>
#include <utility>
#include <cstddef>

template <typename T>
class Stack {
private:
    std::vector<T> data_;
    std::size_t maxLen_;

public:
    // Construct with a maximum capacity
    explicit Stack(std::size_t stackSize)
        : data_(), maxLen_(stackSize)
    {
        // Reserve capacity without increasing size; no default-constructing elements
        data_.reserve(maxLen_);
    }

    // Default copy/move semantics are correct and deep-copy via std::vector
    Stack(const Stack&) = default;
    Stack& operator=(const Stack&) = default;
    Stack(Stack&&) noexcept = default;
    Stack& operator=(Stack&&) noexcept = default;

    ~Stack() = default;

    // Push with strong exception safety: no state change if assignment/constructor throws
    void push(const T &val) {
        if (data_.size() >= maxLen_) {
            throw std::overflow_error("Stack overflow");
        }
        data_.push_back(val); // push_back provides strong exception safety
    }

    // Overload for move
    void push(T &&val) {
        if (data_.size() >= maxLen_) {
            throw std::overflow_error("Stack overflow");
        }
        data_.push_back(std::move(val));
    }

    // Pop throws on underflow to avoid ambiguous sentinel values
    T pop() {
        if (data_.empty()) {
            throw std::underflow_error("Stack underflow");
        }
        T top = std::move(data_.back());
        data_.pop_back();
        return top;
    }

    bool empty() const noexcept { return data_.empty(); }
    std::size_t size() const noexcept { return data_.size(); }
    std::size_t capacity() const noexcept { return maxLen_; }
};