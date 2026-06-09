#pragma once

#include <cstddef>
#include <vector>
#include <utility>

namespace vega {

template<typename T>
class Partition {
public:
    Partition() : id_(0) {}

    Partition(size_t id, std::vector<T> data)
        : id_(id), data_(std::move(data)) {}

    size_t id() const { return id_; }
    size_t size() const { return data_.size(); }
    bool empty() const { return data_.empty(); }

    const std::vector<T>& data() const { return data_; }
    std::vector<T>& data() { return data_; }

    auto begin() { return data_.begin(); }
    auto end() { return data_.end(); }
    auto begin() const { return data_.begin(); }
    auto end() const { return data_.end(); }

private:
    size_t id_;
    std::vector<T> data_;
};

} // namespace vega
