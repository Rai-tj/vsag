#pragma once

#include "quantization/quantizer_parameter.h"
#include "typing.h"
#include "utils/pointer_define.h"

namespace vsag {

// 宏定义智能指针（完全对齐Sparse规范）
DEFINE_POINTER2(MRPQuantizerParam, MRPQuantizerParameter);

static const std::string MRP_DOC_PRUNE_RATIO = "doc_prune_ratio";

class MRPQuantizerParameter : public QuantizerParameter {
public:
    // 默认剪枝比例 0.2f（可保留80%权重）
    float doc_prune_ratio = 0.0f; // 修改

    MRPQuantizerParameter() : QuantizerParameter(QUANTIZATION_TYPE_VALUE_MRP) {
    }

    ~MRPQuantizerParameter() override = default;

    void
    FromJson(const JsonType& json) override {
        // 从JSON读取剪枝比例参数
        if (json.Contains(MRP_DOC_PRUNE_RATIO)) {
            doc_prune_ratio = json[MRP_DOC_PRUNE_RATIO].GetFloat();
        }
    }

    JsonType
    ToJson() const override {
        JsonType json;
        // 写入类型名
        json[TYPE_KEY].SetString(this->GetTypeName());
        // 写入剪枝比例参数
        json[MRP_DOC_PRUNE_RATIO].SetFloat(doc_prune_ratio);
        return json;
    }
};

}  // namespace vsag
