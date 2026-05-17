#pragma once

// 1. 先包含项目基础头文件
#include <cmath>

#include "index_common_param.h"
#include "inner_string_params.h"
#include "quantization/quantizer.h"
#include "quantization/quantizer_parameter.h"
// 2. 【关键】直接包含完整的参数头文件，让编译器看到完整类定义！
#include "quantization/mrp_quantizer_parameter.h"
#include "vsag/dataset.h"
#include "quantization/sparse_quantization/sparse_quantizer.h"

namespace vsag {

    template <MetricType metric = MetricType::METRIC_TYPE_IP>
    class MRPQuantizer : public Quantizer<MRPQuantizer<metric>> {
    public:
        // 构造函数
        explicit MRPQuantizer(const QuantizerParamPtr& param, const IndexCommonParam& common_param);
        explicit MRPQuantizer(Allocator* allocator);

        bool
            TrainImpl(const DataType* data, uint64_t count);

        bool
            EncodeOneImpl(const DataType* data, uint8_t* codes) const;

        bool
            EncodeBatchImpl(const DataType* data, uint8_t* codes, uint64_t count);

        bool
            DecodeOneImpl(const uint8_t* codes, DataType* data);

        bool
            DecodeBatchImpl(const uint8_t* codes, DataType* data, uint64_t count);

        float
            ComputeImpl(const uint8_t* codes1, const uint8_t* codes2) const;

        void
            ProcessQueryImpl(const DataType* query, Computer<MRPQuantizer>& computer) const;

        void
            ComputeDistImpl(Computer<MRPQuantizer>& computer, const uint8_t* codes, float* dists) const;

        void
            ScanBatchDistImpl(Computer<MRPQuantizer<metric>>& computer,
                uint64_t count,
                const uint8_t* codes,
                float* dists) const;

        void
            ReleaseComputerImpl(Computer<MRPQuantizer<metric>>& computer) const;

        void
            SerializeImpl(StreamWriter& writer);

        void
            DeserializeImpl(StreamReader& reader);

        [[nodiscard]] std::string
            NameImpl() const {
            return "mrp";
        }

    private:
        // 保存参数（必须定义）
        QuantizerParamPtr param_;
    };

    // -----------------------------------------------------------------------------
    // 构造函数：保存参数
    // -----------------------------------------------------------------------------
    template <MetricType metric>
    MRPQuantizer<metric>::MRPQuantizer(const QuantizerParamPtr& param,
        const IndexCommonParam& common_param)
        : MRPQuantizer<metric>(common_param.allocator_.get()) {
        this->dim_ = common_param.dim_;
        this->is_trained_ = true;
        this->metric_ = metric;
        this->param_ = param;

        float prune_ratio = 0.0f;
        auto mrp_param = std::dynamic_pointer_cast<MRPQuantizerParameter>(param);
        if (mrp_param) {
            prune_ratio = mrp_param->doc_prune_ratio;
        }
        uint32_t expected_dim = static_cast<uint32_t>(
            std::ceil(static_cast<float>(common_param.dim_) * (1.0f - prune_ratio) + 1e-6f));
        this->code_size_ = sizeof(uint32_t) + expected_dim * sizeof(BufferEntry);
    }

    template <MetricType metric>
    MRPQuantizer<metric>::MRPQuantizer(Allocator* allocator)
        : Quantizer<MRPQuantizer<metric>>(0, allocator) {
        this->metric_ = metric;
        this->is_trained_ = false;
        this->param_ = nullptr;
    }

    // -----------------------------------------------------------------------------
    // 基础接口实现
    // -----------------------------------------------------------------------------
    template <MetricType metric>
    void
        MRPQuantizer<metric>::DeserializeImpl(StreamReader& reader) {
    }

    template <MetricType metric>
    void
        MRPQuantizer<metric>::SerializeImpl(StreamWriter& writer) {
    }

    template <MetricType metric>
    void
        MRPQuantizer<metric>::ReleaseComputerImpl(Computer<MRPQuantizer<metric>>& computer) const {
        this->allocator_->Deallocate(computer.buf_);
    }

    template <MetricType metric>
    void
        MRPQuantizer<metric>::ScanBatchDistImpl(Computer<MRPQuantizer<metric>>& computer,
            uint64_t count,
            const uint8_t* codes,
            float* dists) const {
        throw VsagException(ErrorType::INTERNAL_ERROR, "ScanBatchDistImpl not supported for MRP");
    }

    template <MetricType metric>
    void
        MRPQuantizer<metric>::ComputeDistImpl(Computer<MRPQuantizer>& computer,
            const uint8_t* codes,
            float* dists) const {
        dists[0] = ComputeImpl(computer.buf_, codes);
    }

    template <MetricType metric>
    void
        MRPQuantizer<metric>::ProcessQueryImpl(const DataType* query,
            Computer<MRPQuantizer>& computer) const {
        const auto* sparse_query = reinterpret_cast<const SparseVector*>(query);
        computer.buf_ = reinterpret_cast<uint8_t*>(this->allocator_->Allocate(
            sizeof(uint32_t) + sparse_query->len_ * sizeof(BufferEntry)));
        EncodeOneImpl(query, computer.buf_);
    }

    template <MetricType metric>
    float
        MRPQuantizer<metric>::ComputeImpl(const uint8_t* codes1, const uint8_t* codes2) const {
        const uint32_t len1 = *reinterpret_cast<const uint32_t*>(codes1);
        const auto* entries1 = reinterpret_cast<const BufferEntry*>(codes1 + sizeof(uint32_t));
        const uint32_t len2 = *reinterpret_cast<const uint32_t*>(codes2);
        const auto* entries2 = reinterpret_cast<const BufferEntry*>(codes2 + sizeof(uint32_t));

        float ip = 0.0f;
        uint32_t i = 0, j = 0;
        while (i < len1 && j < len2) {
            if (entries1[i].id < entries2[j].id) {
                i++;
            }
            else if (entries1[i].id > entries2[j].id) {
                j++;
            }
            else {
                ip += entries1[i].val * entries2[j].val;
                i++;
                j++;
            }
        }
        return 1.0f - ip;
    }

    template <MetricType metric>
    bool
        MRPQuantizer<metric>::DecodeBatchImpl(const uint8_t* codes, DataType* data, uint64_t count) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "Decode not supported for MRP");
    }

    template <MetricType metric>
    bool
        MRPQuantizer<metric>::DecodeOneImpl(const uint8_t* codes, DataType* data) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "Decode not supported for MRP");
    }

    template <MetricType metric>
    bool
        MRPQuantizer<metric>::EncodeBatchImpl(const DataType* data, uint8_t* codes, uint64_t count) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "Batch Encode not supported for MRP");
    }

    // -----------------------------------------------------------------------------
    // 核心：Encode + 剪枝（参数100%生效，无任何类型错误）
    // -----------------------------------------------------------------------------
/*    template <MetricType metric>
    bool
        MRPQuantizer<metric>::EncodeOneImpl(const DataType* data, uint8_t* codes) const {
        const SparseVector& sv = *reinterpret_cast<const SparseVector*>(data);
        std::vector<std::pair<uint32_t, float>> terms;
        terms.reserve(sv.len_);

        for (uint32_t i = 0; i < sv.len_; ++i) {
            terms.emplace_back(sv.ids_[i], sv.vals_[i]);
        }

        // 读取参数：现在类是完整的，强转/访问成员完全合法
        float prune_ratio = 0.0f;
        if (this->param_) {
            auto mrp_param = std::dynamic_pointer_cast<MRPQuantizerParameter>(this->param_);
            if (mrp_param) {
                prune_ratio = mrp_param->doc_prune_ratio;
            }
            else {
                throw VsagException(ErrorType::INVALID_ARGUMENT,
                    "MRPQuantizerParameter missing");
            }
        }

        // 执行剪枝
        if (prune_ratio > 0.0f && terms.size() > 1) {
            std::sort(terms.begin(), terms.end(), [](const auto& a, const auto& b) {
                return fabs(a.second) > fabs(b.second);
                });
            size_t keep = std::max(1.0f, terms.size() * (1.0f - prune_ratio));
            terms.resize(keep);
        }

        // 按ID排序
        std::sort(terms.begin(), terms.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
            });

        // 编码输出
        uint32_t final_len = terms.size();
        memcpy(codes, &final_len, sizeof(uint32_t));
        BufferEntry* entries = reinterpret_cast<BufferEntry*>(codes + sizeof(uint32_t));
        for (uint32_t i = 0; i < final_len; ++i) {
            entries[i].id = terms[i].first;
            entries[i].val = terms[i].second;
        }

        return true;
    }
*/
    template <MetricType metric>
    bool
        MRPQuantizer<metric>::EncodeOneImpl(const DataType* data, uint8_t* codes) const {
        const SparseVector& sv = *reinterpret_cast<const SparseVector*>(data);
        std::vector<std::pair<uint32_t, float>> terms;
        terms.reserve(sv.len_);

        for (uint32_t i = 0; i < sv.len_; ++i) {
            terms.emplace_back(sv.ids_[i], sv.vals_[i]);
        }

        // 读取剪枝参数
        float prune_ratio = 0.0f;
        if (this->param_) {
            auto mrp_param = std::dynamic_pointer_cast<MRPQuantizerParameter>(this->param_);
            if (mrp_param) {
                prune_ratio = mrp_param->doc_prune_ratio;
            }
            else {
                throw VsagException(ErrorType::INVALID_ARGUMENT, "MRPQuantizerParameter missing");
            }
        }

        // 计算保留比例（和样例代码的 doc_retain_ratio_ 完全一致）
        float retain_ratio = 1.0f - prune_ratio;

        // ===================== MRP 核心剪枝逻辑（无 goto，对齐论文+样例） =====================
        // 仅当需要剪枝时，执行核心逻辑
        if (retain_ratio < 1.0f && terms.size() > 1) {
            // 1. 按权重绝对值降序排序（MRP 核心）
            std::sort(terms.begin(), terms.end(), [](const auto& a, const auto& b) {
                return fabs(a.second) > fabs(b.second);
                });

            // 2. 计算总权重质量（Total Mass）→ 完全对标样例代码
            float total_mass = 0.0f;
            for (const auto& pair : terms) {
                total_mass += fabs(pair.second);
            }

            // 3. 计算需要保留的目标权重和
            float target_mass = total_mass * retain_ratio;
            float current_mass = 0.0f;
            size_t keep_num = 0;

            // 4. 累加权重，达到阈值即截断（论文证明：召回率更高）
            while (current_mass < target_mass && keep_num < terms.size()) {
                current_mass += fabs(terms[keep_num].second);
                keep_num++;
            }

            // 5. 工程保护：强制保留至少 1 个维度
            keep_num = std::max(keep_num, 1ul);
            terms.resize(keep_num);
        }

        // 统一按维度ID升序排序（稀疏向量计算硬性要求）
        std::sort(terms.begin(), terms.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
            });

        // 编码输出
        uint32_t final_len = terms.size();
        memcpy(codes, &final_len, sizeof(uint32_t));
        BufferEntry* entries = reinterpret_cast<BufferEntry*>(codes + sizeof(uint32_t));
        for (uint32_t i = 0; i < final_len; ++i) {
            entries[i].id = terms[i].first;
            entries[i].val = terms[i].second;
        }

        return true;
    }
    template <MetricType metric>
    bool
        MRPQuantizer<metric>::TrainImpl(const DataType* data, uint64_t count) {
        this->is_trained_ = true;
        return true;
    }

}  // namespace vsag
