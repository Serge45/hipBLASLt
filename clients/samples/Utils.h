#pragma once
#include <bitset>
#include <ios>
#include <ostream>
#include <vector>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <boost/move/utility_core.hpp>
constexpr auto BYTE_SIZE = 8u;

using CUMask = std::vector<uint32_t>;
std::ostream &operator<<(std::ostream &s, const CUMask &mask) {
    s << std::hex;

    for (auto i : mask) {
        s << i << ", ";
    }

    s << std::dec;

    return s;
}

template<std::size_t NumCUs>
CUMask bitsetToCUMask(const std::bitset<NumCUs> &bits) {
    const auto numBits = bits.size();
    constexpr auto numBitsPerElem = sizeof(typename CUMask::value_type) * BYTE_SIZE;
    auto numElemRequired = numBits / numBitsPerElem;

    if (numBits % numBitsPerElem) {
        ++numElemRequired;
    }

    CUMask mask(numElemRequired, 0);

    for (std::size_t i = 0; i < numBits; ++i) {
        const auto elemIdx = i / numBitsPerElem;
        const auto bitIdx = i % numBitsPerElem;
        auto &maskElem = mask.at(elemIdx);

        if (bits.test(i)) {
            maskElem |= (1 << bitIdx);
        }
    }

    std::cout << mask << '\n';

    return mask;
}

template<std::size_t NumCUs>
std::vector<std::bitset<NumCUs>> groupSizesToCUMasks(const std::vector<std::size_t> &groupSizes) {
    const auto numBits = std::accumulate(begin(groupSizes), end(groupSizes), 0ull);

    std::vector<std::bitset<NumCUs>> masks(groupSizes.size());
    std::size_t bitIdx{};

    for (std::size_t i = 0; i < groupSizes.size(); ++i) {
        for (std::size_t j = 0; j < groupSizes[i]; ++j) {
            masks[i].set(bitIdx++);
        }
    }

    return masks;
}

template<std::size_t NumCUs>
std::bitset<NumCUs> generateSEDistributedBitsets(std::size_t group, std::size_t numCUs, std::size_t numCUsPerGroup) {
    std::bitset<NumCUs> bits;

    for (std::size_t j = 0; j < numCUs; ++j) {
        bits.set(group * numCUsPerGroup + j);
    }

    std::cout << "Creating CU mask: " << bits.to_string() << '\n';
    return bits;
}

template<std::size_t NumCUs, std::size_t NumSEs, std::size_t NumCUsPerSE>
std::bitset<NumCUs> generateSEPackedBitsets(std::size_t numCUs, std::size_t &curSEIdx, std::size_t &curSEUsage) {
    std::bitset<NumCUs> bits;

    for (std::size_t i = 0; i < numCUs; ++i) {
        std::size_t realUsage{};

        if (curSEUsage < NumCUsPerSE) {
            const auto bitIdx = curSEUsage * NumSEs + curSEIdx;

            if (bitIdx < bits.size()) {
                bits.set(bitIdx);
                ++curSEUsage;
            } else {
                //reset since CUs on current SE are all used.
                realUsage = curSEUsage;
                curSEUsage = NumCUsPerSE;
                --i;
            }
        }

        if (curSEUsage == NumCUsPerSE) {
            std::cout << "SE" << curSEIdx << ":" << realUsage << '\n';
            ++curSEIdx;
            curSEUsage = 0;
        }
    }

    std::cout << "Creating CU mask: " << bits.to_string() << '\n';
    return bits;
}

template<std::size_t NumCUs, std::size_t NumSEs, std::size_t NumCUsPerSE = 16ull>
std::vector<CUMask> generateCUMaskGroups(const std::vector<std::size_t> &groupSizes, bool packSE) {
    std::vector<std::bitset<NumCUs>> rawMasks;

    if (packSE) {
        std::size_t curSEIdx{};
        std::size_t curSEUsage{};

        for (auto s : groupSizes) {
            rawMasks.push_back(generateSEPackedBitsets<NumCUs, NumSEs, NumCUsPerSE>(s, curSEIdx, curSEUsage));
        }
    } else {
        std::size_t offset{};

        for (std::size_t i = 0; i < groupSizes.size(); ++i) {
            std::bitset<NumCUs> bits;

            for (std::size_t j = 0; j < groupSizes[i]; ++j) {
                bits.set(j + offset);
            }

            rawMasks.push_back(bits);
            offset += groupSizes[i];
            std::cout << "Creating CU mask: " << bits.to_string() << '\n';
        }
    }

    assert(rawMasks.size() == groupSizes.size());

    std::vector<CUMask> masks;

    std::transform(begin(rawMasks), end(rawMasks), std::back_inserter(masks), [](auto &bits) {
        return bitsetToCUMask(bits);
    });

    return masks;
}


template<std::size_t NumCUs>
std::vector<CUMask> generateCUMaskGroups(std::size_t numGroups) {
    std::vector<std::bitset<NumCUs>> rawMasks;
    const auto numCUsPerGroup = NumCUs / numGroups;
    const auto numCUsLastGroup = numCUsPerGroup + (NumCUs % numCUsPerGroup);

    for (std::size_t i = 0; i + 1 < numGroups; ++i) {
        auto bits = generateSEDistributedBitsets<NumCUs>(i, numCUsPerGroup, numCUsPerGroup);
        rawMasks.push_back(bits);
    }

    rawMasks.push_back(generateSEDistributedBitsets<NumCUs>(numGroups - 1, numCUsLastGroup, numCUsPerGroup));

    assert(rawMasks.size() == numGroups);

    std::vector<CUMask> masks;

    std::transform(begin(rawMasks), end(rawMasks), std::back_inserter(masks), [](auto &bits) {
        return bitsetToCUMask(bits);
    });

    return masks;
}

template<std::size_t NumCUs, std::size_t NumSEs, std::size_t NumCUsPerSE = 16ull>
std::vector<CUMask> generateSEPackedCUMaskGroups(std::size_t numGroups) {
    static_assert(NumCUs <= NumSEs * NumCUsPerSE, "Invalid CU & SE setting");
    std::vector<std::bitset<NumCUs>> rawMasks;
    const auto numCUsPerGroup = NumCUs / numGroups;
    const auto numCUsLastGroup = numCUsPerGroup + (NumCUs % numCUsPerGroup);
    std::size_t curSEIdx{};
    std::size_t curSEUsage{};

    for (std::size_t i = 0; i + 1 < numGroups; ++i) {
        auto bits = generateSEPackedBitsets<NumCUs, NumSEs, NumCUsPerSE>(numCUsPerGroup, curSEIdx, curSEUsage);
        rawMasks.push_back(bits);
        //prevent fron SE overlapping
        // if (curSEUsage) {
        //     curSEUsage = 0;
        //     ++curSEIdx;
        // }
    }

    rawMasks.push_back(generateSEPackedBitsets<NumCUs, NumSEs, NumCUsPerSE>(numCUsLastGroup, curSEIdx, curSEUsage));
    assert(curSEIdx <= NumSEs && "Invalid SEIdx");
    assert(rawMasks.size() == numGroups);

    std::vector<CUMask> masks;

    std::transform(begin(rawMasks), end(rawMasks), std::back_inserter(masks), [](auto &bits) {
        return bitsetToCUMask(bits);
    });

    return masks;
}

template<typename T>
std::vector<std::vector<T>> split(const std::vector<T> &vec, const std::vector<std::size_t> &chunks) {
    std::size_t offset{};
    std::vector<std::vector<T>> ret;

    for (auto c : chunks) {
        std::vector<T> chunk;
        auto beg = std::next(begin(vec), offset);
        auto end = std::next(begin(vec), offset + c);
        std::copy(beg, end, std::back_inserter(chunk));
        ret.push_back(chunk);
        offset += c;
    }

    return ret;
}

template<typename T>
std::vector<std::vector<T>> split(const std::vector<T> &vec, std::size_t numChunks) {
    std::vector<std::vector<T>> chunks;
    std::size_t chunkSize = vec.size() / numChunks;
    std::size_t alignedSize = numChunks * chunkSize;
    std::size_t cursor{};

    for (; cursor < alignedSize; cursor += chunkSize) {
        std::vector<T> chunk{begin(vec) + cursor, begin(vec) + cursor + chunkSize};
        chunks.push_back(chunk);
    }

    std::vector<T> remainders{begin(vec) + cursor, begin(vec) + vec.size()};
    std::copy(begin(remainders), end(remainders), std::back_inserter(chunks.back()));

    assert(numChunks == chunks.size());
    return chunks;
}

struct SafeEvent {
    SafeEvent() {
        hipEventCreate(&event);
    }

    ~SafeEvent() {
        hipEventDestroy(event);
    }

    SafeEvent(const SafeEvent &) = delete;
    SafeEvent(SafeEvent &&) = delete;
    SafeEvent &operator=(const SafeEvent &) = delete;
    SafeEvent &operator=(SafeEvent &&) = delete;

    operator hipEvent_t() const {
        return event;
    }

private:
    hipEvent_t event;
};

struct SafeEventDurCalculator {
    static float calculate(const SafeEvent &a, const SafeEvent &b) {
        float t{};
        hipEventElapsedTime(&t, a, b);
        return t;
    }
};

template<typename T>
struct SafeHipBuffer {
    explicit SafeHipBuffer(std::size_t numElem) {
        if (numElem) {
            numBytes = numElem * sizeof(T);
            (void)hipMalloc(&buffer, numBytes);
        }
    }

    SafeHipBuffer(const SafeHipBuffer &) = delete;
    SafeHipBuffer(SafeHipBuffer &&rhs) {
        this->~SafeHipBuffer();
        this->buffer = rhs.buffer;
        rhs.buffer = nullptr;
    }

    SafeHipBuffer &operator=(const SafeHipBuffer &) = delete;
    SafeHipBuffer &operator=(SafeHipBuffer &&rhs) {
        this->~SafeHipBuffer();
        this->buffer = rhs.buffer;
        rhs.buffer = nullptr;
        return *this;
    }

    ~SafeHipBuffer() {
        if (buffer) {
            hipFree(buffer);
        }
    }

    operator void*() {
        return buffer;
    }

    std::size_t bufferSize() const {
        return numBytes;
    }

private:
    T *buffer{};
    std::size_t numBytes{};
};

struct SafeMatmulPreference {
    explicit SafeMatmulPreference(uint64_t workspaceNumBytes = 32 * 1024 * 1024)
    : buffer(workspaceNumBytes) {
    
    hipblasLtMatmulPreferenceCreate(&pref);
    hipblasLtMatmulPreferenceSetAttribute(
        pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceNumBytes, sizeof(workspaceNumBytes));
    }

    ~SafeMatmulPreference() {
        if (pref) {
            hipblasLtMatmulPreferenceDestroy(pref);
            pref = nullptr;
        }
    }

    SafeMatmulPreference(BOOST_RV_REF(SafeMatmulPreference) rhs) 
    : buffer(0ull) {
        buffer = std::move(rhs.buffer);
        pref = rhs.pref;
        rhs.pref = nullptr;
    }

    SafeMatmulPreference &operator=(BOOST_RV_REF(SafeMatmulPreference) rhs) {
        if (buffer) {
            buffer.~SafeHipBuffer();
        }

        buffer = std::move(rhs.buffer);
        pref = rhs.pref;
        rhs.pref = nullptr;
        return *this;
    }

    operator hipblasLtMatmulPreference_t() {
        return pref;
    }

    SafeHipBuffer<uint8_t> &getBuffer() {
        return buffer;
    }

private:
    BOOST_MOVABLE_BUT_NOT_COPYABLE(SafeMatmulPreference)

private:
    SafeHipBuffer<uint8_t> buffer;
    hipblasLtMatmulPreference_t pref;
};