#pragma once

#include <vector>
#include <algorithm>
#include <cassert>

// Creates a vector that accounts for the start offset
template <typename T>
class RangedVector
{
    public:
        RangedVector() : data_(), start_(0), end_(0) {}

        RangedVector(int start, int end, T v = T()) 
            : data_(end-start, v), start_(start), end_(end) {}

        void resize(int start, int end , T v = T())
        {
            data_.resize(end-start, v);
            start_ = start;
            end_ = end;
        }

        void clear()
        {
            data_.clear();
        }

        T& operator[](int idx)
        {
            return data_[idx-start_];
        }

        const T& operator[](int idx) const
        {
            return data_[idx-start_];
        }

    private:
        std::vector<T> data_;
        int start_;
        int end_;
};

template <typename T>
class TriMatrix
{
    public:
        TriMatrix() : data_() {}

        TriMatrix(int sz, T v = T(), int diag=0) : data_(sz)
        {
            for (auto i=0; i!=sz; ++i)
                data_[i] = std::move(RangedVector<T>(i+diag, sz, v));
        }

        void resize(int sz, T v = T(), int diag=0)
        {
            data_.resize(sz);
            for (auto i=0; i!=sz; ++i)
                data_[i].resize(i+diag, sz, v);
        }

        void clear()
        {
            data_.clear();
        }

        size_t size() const { return data_.size(); }

        RangedVector<T>& operator[](size_t idx) { return data_[idx]; }
        const RangedVector<T>& operator[](size_t idx) const { return data_[idx]; }

    private:
        std::vector< RangedVector<T> > data_;
};

/**
// Upper triangular matrix but with contiguous memory and no reallocations (assumes that max sz <= 501)
template <typename T>
class TriMatrixNew
{
    public:
        TriMatrix() : data_(), sz_(0), offsets() {}

        TriMatrix(int sz, T v = T()) : data_(sz * (sz + 1) / 2, v), sz_(0) {
            offsets.resize(sz);

            offsets[0] = 0;
            for (uint32_t i = 0; i < sz; ++i) {
                offsets[i + 1] = offsets[i] + (sz - i);
            }
        }

        void resize(int sz, T v = T())
        {

            std::fill(data_.begin(), data_.end(), v);
            sz_ = sz;
        }

        size_t size() const { return sz_; }

        T& operator()(size_t i, size_t j) {
            assert(j < sz_ && i <= j);
            return data_[offsets[501 - sz + i] + j - i];
        }

        const T& operator()(size_t i, size_t j) const {
            assert(j < sz_ && i <= j);
            return data_[offsets[501 - sz + i] + j - i];
        }

    private:
        std::vector<T> data_;
        size_t sz_;
        std::vector<uint32_t> offsets;
};
*/