# Beauty D1 数据适配协议

`reclab_agent.beauty.import_beauty_bundle` 只负责把已经划分好的 D1 文件接入 RecLab，不重新过滤、划分或选择用户。它要求源文件使用 schema 2、`sasrec_source_order_train_catalog` 协议，并要求调用者提供预先冻结的源文件字节 SHA256。

## 保留什么

- 商品目录必须按 ID 连续排列为 1…N，且每件商品至少出现在一条训练交互中。不能把冷启动验证或测试商品加入目录。
- 所有源用户的训练历史均保留，包含重复交互及完整原始顺序。模型可截断输入，但评分时的 seen mask 使用完整历史。
- 验证人群严格为源文件 `valid > 0` 的全部用户。先确定验证人群、公开文件和公开身份，再进入独立的测试数据处理步骤。
- `valid == 0` 根据 `valid_raw_item_id` 是否为空，分别计为“目标不在训练目录”或“无留出目标”。不从测试结果、目标或可用性中选择验证人群。
- 正数目标在目录外、目标已在历史中、计数不一致、目录多出未训练商品等情况均直接报错，不能通过删除用户或修改历史“修复”。
- 测试人群独立取 `test > 0`，不与验证人群求交集；测试历史为完整 train，加上可用的暖验证目标。`valid == 0` 时不追加商品 0。

这与原 `semantic_id_rec.metrics.evaluation_records` 的暖目标人群、历史构造和重复目标报错规则一致。原始数据的源频次单次过滤、同时间戳稳定排序等预处理约定由 D1 文件和其来源哈希固定；适配器不会重新执行这些步骤。

## 公开与私有文件

候选执行器只能拿到 `public/`，其中恰有三个文件：

| 文件 | 内容 |
|---|---|
| `public/manifest.json` | 目录、训练/验证计数、验证排除原因、公开内容身份、固定协议和原始 reviews/metadata 来源哈希 |
| `public/train.json` | 全部用户的完整训练历史 |
| `public/valid.json` | 原暖验证用户的 history/target，用于开发、验证及实验选择 |
| `private/test.json` | 独立暖测试用户的 history/target，只供受信评分器使用 |
| `private/provenance.json` | 完整准备文件字节哈希、源统计、目录元数据哈希、测试人群/排除计数/内容哈希 |

公开验证目标是开发集信息，允许实验过程使用；它不构成独立的最终测试证据。原始完整 JSON 绝不能作为候选挂载的数据文件，因为其中包含测试目标。目录文本元数据目前只记录来源内容哈希，尚未导出为候选 RAG 语料。

源准备文件的 SHA256 是“原始文件字节身份”，`bundle.data_hash` 是规范化公开 manifest/train/valid 的身份，两者数值不同。改变私有测试值会改变准备文件哈希和私有 provenance，但不会改变公开验证人群、公开文件字节、数据 ID 或 `data_hash`。公开身份同时绑定原始 reviews/metadata 来源哈希；若更换了原始数据来源，应产生新的公开身份。

## 使用方式

从仓库根目录运行：

```powershell
.\.venv\Scripts\python.exe -m reclab_agent.beauty `
  --source .reclab/cloud-source/beauty-sasrec.json `
  --destination .reclab/datasets/beauty-d1 `
  --expected-sha256 7a2dae80bf7053c91d4eda79f33ff997df45c07c1fd968f36d77b591ed828efb `
  --expected-valid-users 22312 `
  --expected-catalog-items 12068
```

上面哈希和计数是待导入 D1 的冻结期望，命令示例本身不说明真实导入、评分或训练已经成功。成功后返回 `DatasetBundle.info()`，并把详细来源保存在私有 provenance。已有目的目录（包括空目录或中断留下的目录）一律拒绝覆盖；重试时选择新目录，避免把不同版本混在一起。

## 如何验证

`tests/test_beauty.py` 包含真实导入前可运行的协议测试：训练/验证/测试人群各自保留、冷目标排除原因、测试修改不影响公开身份、重复目标直接失败、非法历史/目录/计数拒绝、冻结哈希和期望规模校验、重复 JSON 键拒绝、禁止覆盖，以及独立评分器接受导出的暖验证预测。

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_beauty -v
```

单元测试使用五商品的构造数据，它们证明适配合同的行为；真实 Beauty 导入及基线复现必须另外记录运行结果。public/private 目录分离也只定义挂载边界，不能代替操作系统或容器对测试标签访问的实际隔离验证。
