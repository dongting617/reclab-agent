# 模型优化实验报告

**状态：已结束，结果不完整**

本轮记录的最佳候选为 ReAct · candidate-4：ndcg@20 从 0.080042 提高到 0.081267，绝对变化 +0.001225（相对 +1.53%），保护指标达标。 各模式有效候选数量不完整，不宣布模式胜者。

## 实验范围

- **任务**：相同预算下的三种 Agent 模式模型优化实验
- **模型**：SASRec
- **Agent 模式**：模式对比实验
- **数据集**：beauty-d1-e2c0b12d23d8e77b
- **训练种子**：2026

## 实测结果

| 候选 | ndcg@20 | recall@20 | 主指标变化（绝对 / 相对） | 判定 |
| --- | --- | --- | --- | --- |
| 基线 · sasrec-b1-retrain | 0.080042 | 0.154222 | — | 基线 |
| ReAct · candidate-1 | — | — | — | 无有效评分 |
| ReAct · candidate-2 | — | — | — | 无有效评分 |
| ReAct · candidate-3 | — | — | — | 无有效评分 |
| ReAct · candidate-4 | 0.081267 | 0.155835 | +0.001225 / +1.53% | 优于基线且保护指标达标 |
| ReAct · candidate-5 | — | — | — | 无有效评分 |
| Plan and Execute · candidate-1 | — | — | — | 无有效评分 |
| Plan and Execute · candidate-3 | — | — | — | 无有效评分 |
| Plan and Execute · candidate-5 | 0.08057 | 0.157314 | +0.000529 / +0.66% | 优于基线且保护指标达标 |
| Reflection · candidate-1 | — | — | — | 无有效评分 |
| Reflection · candidate-2 | 0.081188 | 0.154984 | +0.001146 / +1.43% | 优于基线且保护指标达标 |
| Reflection · candidate-3 | 0.080722 | 0.154625 | +0.000681 / +0.85% | 未超过已记录最佳主指标 |
| Reflection · candidate-4 | 0.080715 | 0.155567 | +0.000673 / +0.84% | 未超过已记录最佳主指标 |

失败或缺失评分以 — 表示，不按零分参与比较。保护指标要求不低于基线。

## 最佳记录与产物

**ReAct · candidate-4**

[训练记录](../../.reclab/sc/sessions/20261120T020207Z-compare-eeddcd55/strategies/react/experiments/exp-283754078c08998409c76ef38d8ba51c/train/training.json) · [训练日志](../../.reclab/sc/sessions/20261120T020207Z-compare-eeddcd55/strategies/react/experiments/exp-283754078c08998409c76ef38d8ba51c/train-log.json) · [模型权重](../../.reclab/sc/sessions/20261120T020207Z-compare-eeddcd55/strategies/react/experiments/exp-283754078c08998409c76ef38d8ba51c/train/best.pt) · [评分证据](../../.reclab/sc/sessions/20261120T020207Z-compare-eeddcd55/strategies/react/experiments/exp-283754078c08998409c76ef38d8ba51c/grade.json) · [源码 · configs/default.json](../../.reclab/sc/sessions/20261120T020207Z-compare-eeddcd55/strategies/react/snapshots/c7553d42a06cae1c5ab296f3a08f8894156435c3f4439a61d0d0bcde7fc49dab/configs/default.json) · [源码 · src/sasrec.py](../../.reclab/sc/sessions/20261120T020207Z-compare-eeddcd55/strategies/react/snapshots/c7553d42a06cae1c5ab296f3a08f8894156435c3f4439a61d0d0bcde7fc49dab/src/sasrec.py)

## 执行情况与资源

实际训练尝试合计 13 次（含基线）。

- **ReAct**：训练尝试 5 次；可用评分 1 个；无可用评分 4 个；模型轮次 42 次；记录 tokens 323,875；服务器训练/预测 252.1 秒。
- **Plan and Execute**：训练尝试 3 次；可用评分 1 个；无可用评分 2 个；模型轮次 40 次；记录 tokens 333,007；服务器训练/预测 268.2 秒。
  - candidate-2：规划输出未通过格式检查，未启动训练。
  - candidate-4：规划输出未通过格式检查，未启动训练。
- **Reflection**：训练尝试 4 次；可用评分 3 个；无可用评分 1 个；模型轮次 48 次；记录 tokens 512,602；服务器训练/预测 595.7 秒。
  - candidate-5：补丁预算耗尽，重复候选请求被拦截，未启动新训练。

## 候选改动与失败记录

### ReAct · candidate-1

验证预测出现非有限 logits（NaN/Inf），没有取得有效评分；日志未证明梯度爆炸。

configs/default.json（冻结源码快照）：

```diff
--- configs/default.json
+++ configs/default.json
@@ -3,10 +3,10 @@
     "file": "src/sasrec.py",
     "class": "SASRec",
     "kwargs": {
-      "hidden_size": 64,
-      "num_layers": 1,
-      "num_heads": 1,
-      "dropout": 0.4
+      "hidden_size": 128,
+      "num_layers": 2,
+      "num_heads": 4,
+      "dropout": 0.3
     }
   },
   "max_length": 50,

```

产物：[训练日志](../../.reclab/sc/sessions/20261120T020207Z-compare-eeddcd55/strategies/react/experiments/exp-a75c67fe5ae6dad0cc536b8127b8c385/train-log.json) · [源码 · configs/default.json](../../.reclab/sc/sessions/20261120T020207Z-compare-eeddcd55/strategies/react/snapshots/e0cbc027013e6b1470932c9ea968e1133a182dc8a32ab00f0f4cf52e6997736b/configs/default.json) · [源码 · src/sasrec.py](../../.reclab/sc/sessions/20261120T020207Z-compare-eeddcd55/strategies/react/snapshots/e0cbc027013e6b1470932c9ea968e1133a182dc8a32ab00f0f4cf52e6997736b/src/sasrec.py)

### ReAct · candidate-2

验证预测出现非有限 logits（NaN/Inf），没有取得有效评分；日志未证明梯度爆炸。

configs/default.json（冻结源码快照）：

```diff
--- configs/default.json
+++ configs/default.json
@@ -3,10 +3,10 @@
     "file": "src/sasrec.py",
     "class": "SASRec",
     "kwargs": {
-      "hidden_size": 64,
-      "num_layers": 1,
-      "num_heads": 1,
-      "dropout": 0.4
+      "hidden_size": 128,
+      "num_layers": 2,
+      "num_heads": 4,
+      "dropout": 0.1
     }
   },
   "max_length": 50,

```

产物：[训练日志](../../.reclab/sc/sessions/20261120T020207Z-compare-eeddcd55/strategies/react/experiments/exp-1952406884533138a1cf780d4e84af87/train-log.json) · [源码 · configs/default.json](../../.reclab/sc/sessions/20261120T020207Z-compare-eeddcd55/strategies/react/snapshots/2e238cd5f06719d610d14888d71eb7d260e0c0a1dc7b24029a67794c6113990f/configs/default.json) · [源码 · src/sasrec.py](../../.reclab/sc/sessions/20261120T020207Z-compare-eeddcd55/strategies/react/snapshots/2e238cd5f06719d610d14888d71eb7d260e0c0a1dc7b24029a67794c6113990f/src/sasrec.py)

### ReAct · candidate-3

验证预测出现非有限 logits（NaN/Inf），没有取得有效评分；日志未证明梯度爆炸。

configs/default.json（冻结源码快照）：

```diff
--- configs/default.json
+++ configs/default.json
@@ -4,9 +4,9 @@
     "class": "SASRec",
     "kwargs": {
       "hidden_size": 64,
-      "num_layers": 1,
-      "num_heads": 1,
-      "dropout": 0.4
+      "num_layers": 2,
+      "num_heads": 4,
+      "dropout": 0.2
     }
   },
   "max_length": 50,

```

产物：[训练日志](../../.reclab/sc/sessions/20261120T020207Z-compare-eeddcd55/strategies/react/experiments/exp-8b0c28ec72d46ecaccf7a2a470c09405/train-log.json) · [源码 · configs/default.json](../../.reclab/sc/sessions/20261120T020207Z-compare-eeddcd55/strategies/react/snapshots/ead13a1cc948e0c516e1227db8060de71e79ef049d1f0e23287eeea1170cf063/configs/default.json) · [源码 · src/sasrec.py](../../.reclab/sc/sessions/20261120T020207Z-compare-eeddcd55/strategies/react/snapshots/ead13a1cc948e0c516e1227db8060de71e79ef049d1f0e23287eeea1170cf063/src/sasrec.py)

### ReAct · candidate-4

src/sasrec.py（冻结源码快照）：

```diff
--- src/sasrec.py
+++ src/sasrec.py
@@ -33,7 +33,7 @@
             dropout=dropout,
             activation="gelu",
             batch_first=True,
-            norm_first=True,
+            norm_first=False,
         )
         self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
         self.output_norm = nn.LayerNorm(hidden_size)

```

产物：[训练记录](../../.reclab/sc/sessions/20261120T020207Z-compare-eeddcd55/strategies/react/experiments/exp-283754078c08998409c76ef38d8ba51c/train/training.json) · [训练日志](../../.reclab/sc/sessions/20261120T020207Z-compare-eeddcd55/strategies/react/experiments/exp-283754078c08998409c76ef38d8ba51c/train-log.json) · [模型权重](../../.reclab/sc/sessions/20261120T020207Z-compare-eeddcd55/strategies/react/experiments/exp-283754078c08998409c76ef38d8ba51c/train/best.pt) · [评分证据](../../.reclab/sc/sessions/20261120T020207Z-compare-eeddcd55/strategies/react/experiments/exp-283754078c08998409c76ef38d8ba51c/grade.json) · [源码 · configs/default.json](../../.reclab/sc/sessions/20261120T020207Z-compare-eeddcd55/strategies/react/snapshots/c7553d42a06cae1c5ab296f3a08f8894156435c3f4439a61d0d0bcde7fc49dab/configs/default.json) · [源码 · src/sasrec.py](../../.reclab/sc/sessions/20261120T020207Z-compare-eeddcd55/strategies/react/snapshots/c7553d42a06cae1c5ab296f3a08f8894156435c3f4439a61d0d0bcde7fc49dab/src/sasrec.py)

### ReAct · candidate-5

验证预测出现非有限 logits（NaN/Inf），没有取得有效评分；日志未证明梯度爆炸。

configs/default.json（冻结源码快照）：

```diff
--- configs/default.json
+++ configs/default.json
@@ -3,10 +3,10 @@
     "file": "src/sasrec.py",
     "class": "SASRec",
     "kwargs": {
-      "hidden_size": 64,
-      "num_layers": 1,
-      "num_heads": 1,
-      "dropout": 0.4
+      "hidden_size": 128,
+      "num_layers": 2,
+      "num_heads": 4,
+      "dropout": 0.3
     }
   },
   "max_length": 50,

```

产物：[训练日志](../../.reclab/sc/sessions/20261120T020207Z-compare-eeddcd55/strategies/react/experiments/exp-89f220bb1cbab2b214749e8fc8fd85ed/train-log.json) · [源码 · configs/default.json](../../.reclab/sc/sessions/20261120T020207Z-compare-eeddcd55/strategies/react/snapshots/25f5b2ac2d32f4cb326ea32d23d8d27230e54044749678051b6e3ddcb848b372/configs/default.json) · [源码 · src/sasrec.py](../../.reclab/sc/sessions/20261120T020207Z-compare-eeddcd55/strategies/react/snapshots/25f5b2ac2d32f4cb326ea32d23d8d27230e54044749678051b6e3ddcb848b372/src/sasrec.py)

### Plan and Execute · candidate-1

验证预测出现非有限 logits（NaN/Inf），没有取得有效评分；日志未证明梯度爆炸。

configs/default.json（冻结源码快照）：

```diff
--- configs/default.json
+++ configs/default.json
@@ -3,9 +3,9 @@
     "file": "src/sasrec.py",
     "class": "SASRec",
     "kwargs": {
-      "hidden_size": 64,
-      "num_layers": 1,
-      "num_heads": 1,
+      "hidden_size": 128,
+      "num_layers": 2,
+      "num_heads": 4,
       "dropout": 0.4
     }
   },

```

产物：[训练日志](../../.reclab/sc/sessions/20261120T020207Z-compare-eeddcd55/strategies/plan_execute/experiments/exp-b708ae5fa0163b142c2ce8f7a8273fd6/train-log.json) · [源码 · configs/default.json](../../.reclab/sc/sessions/20261120T020207Z-compare-eeddcd55/strategies/plan_execute/snapshots/e69978dff16b80f602b6960da9e566ecd7bad20fe7581331cd8d2d4ce878f3c3/configs/default.json) · [源码 · src/sasrec.py](../../.reclab/sc/sessions/20261120T020207Z-compare-eeddcd55/strategies/plan_execute/snapshots/e69978dff16b80f602b6960da9e566ecd7bad20fe7581331cd8d2d4ce878f3c3/src/sasrec.py)

### Plan and Execute · candidate-3

验证预测出现非有限 logits（NaN/Inf），没有取得有效评分；日志未证明梯度爆炸。

configs/default.json（冻结源码快照）：

```diff
--- configs/default.json
+++ configs/default.json
@@ -3,17 +3,17 @@
     "file": "src/sasrec.py",
     "class": "SASRec",
     "kwargs": {
-      "hidden_size": 64,
-      "num_layers": 1,
-      "num_heads": 1,
+      "hidden_size": 128,
+      "num_layers": 2,
+      "num_heads": 4,
       "dropout": 0.4
     }
   },
   "max_length": 50,
   "epochs": 30,
   "patience": 5,
-  "batch_size": 256,
-  "eval_batch_size": 256,
+  "batch_size": 128,
+  "eval_batch_size": 128,
   "learning_rate": 0.001,
   "weight_decay": 0.0001,
   "grad_clip": 5.0,

```

产物：[训练日志](../../.reclab/sc/sessions/20261120T020207Z-compare-eeddcd55/strategies/plan_execute/experiments/exp-f4e4cb1e2c1bd65305d3b8ef1efbb246/train-log.json) · [源码 · configs/default.json](../../.reclab/sc/sessions/20261120T020207Z-compare-eeddcd55/strategies/plan_execute/snapshots/848acc74476f57ce6c2b69ba496fc5cddbc6ad9c2a9e4b81edc1593264bc7786/configs/default.json) · [源码 · src/sasrec.py](../../.reclab/sc/sessions/20261120T020207Z-compare-eeddcd55/strategies/plan_execute/snapshots/848acc74476f57ce6c2b69ba496fc5cddbc6ad9c2a9e4b81edc1593264bc7786/src/sasrec.py)

### Plan and Execute · candidate-5

configs/default.json（冻结源码快照）：

```diff
--- configs/default.json
+++ configs/default.json
@@ -4,9 +4,9 @@
     "class": "SASRec",
     "kwargs": {
       "hidden_size": 64,
-      "num_layers": 1,
+      "num_layers": 2,
       "num_heads": 1,
-      "dropout": 0.4
+      "dropout": 0.5
     }
   },
   "max_length": 50,

```

产物：[训练记录](../../.reclab/sc/sessions/20261120T020207Z-compare-eeddcd55/strategies/plan_execute/experiments/exp-fc6c310740f68013e0a4e1a6113c93d5/train/training.json) · [训练日志](../../.reclab/sc/sessions/20261120T020207Z-compare-eeddcd55/strategies/plan_execute/experiments/exp-fc6c310740f68013e0a4e1a6113c93d5/train-log.json) · [模型权重](../../.reclab/sc/sessions/20261120T020207Z-compare-eeddcd55/strategies/plan_execute/experiments/exp-fc6c310740f68013e0a4e1a6113c93d5/train/best.pt) · [评分证据](../../.reclab/sc/sessions/20261120T020207Z-compare-eeddcd55/strategies/plan_execute/experiments/exp-fc6c310740f68013e0a4e1a6113c93d5/grade.json) · [源码 · configs/default.json](../../.reclab/sc/sessions/20261120T020207Z-compare-eeddcd55/strategies/plan_execute/snapshots/a88176d9b2ee79df5ef2d5740d4e5296ba76d6fbaeef5a4aee8bfda946a282d0/configs/default.json) · [源码 · src/sasrec.py](../../.reclab/sc/sessions/20261120T020207Z-compare-eeddcd55/strategies/plan_execute/snapshots/a88176d9b2ee79df5ef2d5740d4e5296ba76d6fbaeef5a4aee8bfda946a282d0/src/sasrec.py)

### Reflection · candidate-1

模型执行触发递归深度错误（RecursionError），没有取得有效评分。

src/sasrec.py（冻结源码快照）：

```diff
--- src/sasrec.py
+++ src/sasrec.py
@@ -62,5 +62,19 @@
         return self.output_norm(x[:, -1])

     def catalog_logits(self, user_states: torch.Tensor) -> torch.Tensor:
+        """Original baseline logit computation with reflection enhancement."""
+        # Use reflection-aware logits for better discrimination
+        mixed, gate = self.reflection_logits(user_states)
+        return mixed
+
+    def reflection_logits(self, user_states: torch.Tensor) -> torch.Tensor:
+        """Reflection-aware logits: combine original scores with negation signal."""
+        base_logits = self.catalog_logits(user_states)
+        # Compute mean item representation for negation signal
         catalog = self.item_embedding.weight[1:]
-        return user_states @ catalog.t() + self.item_bias[1:]
+        mean_item = catalog.mean(dim=0, keepdim=True)  # [1, hidden_size]
+        negation_signal = (mean_item - user_states.unsqueeze(1)) @ catalog.t()  # [batch, num_items]
+        # Learnable gate to balance original vs reflection
+        gate = torch.sigmoid(torch.zeros(1, device=user_states.device) * 0.5)
+        mixed_logits = (1.0 - gate) * base_logits + gate * negation_signal
+        return mixed_logits, gate

```

产物：[训练日志](../../.reclab/sc/sessions/20261120T020207Z-compare-eeddcd55/strategies/reflection/experiments/exp-50ece63042d1981aae2dee4d7e5c38b3/train-log.json) · [源码 · configs/default.json](../../.reclab/sc/sessions/20261120T020207Z-compare-eeddcd55/strategies/reflection/snapshots/acb92ecc5955df4a54e38b9eaa5795182e4319ed61eea29fbf5ce06927ec9e7d/configs/default.json) · [源码 · src/sasrec.py](../../.reclab/sc/sessions/20261120T020207Z-compare-eeddcd55/strategies/reflection/snapshots/acb92ecc5955df4a54e38b9eaa5795182e4319ed61eea29fbf5ce06927ec9e7d/src/sasrec.py)

### Reflection · candidate-2

src/sasrec.py（冻结源码快照）：

```diff
--- src/sasrec.py
+++ src/sasrec.py
@@ -38,6 +38,9 @@
         self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
         self.output_norm = nn.LayerNorm(hidden_size)
         self.item_bias = nn.Parameter(torch.zeros(num_items + 1))
+        # Reflection enhancement components
+        self.reflection_fc1 = nn.Linear(hidden_size, hidden_size)
+        self.gate_weight = nn.Parameter(torch.zeros(1))
         self.reset_parameters()

     def reset_parameters(self) -> None:
@@ -62,5 +65,28 @@
         return self.output_norm(x[:, -1])

     def catalog_logits(self, user_states: torch.Tensor) -> torch.Tensor:
+        """Base recommendation logits from item similarity."""
         catalog = self.item_embedding.weight[1:]
         return user_states @ catalog.t() + self.item_bias[1:]
+
+    def reflection_logits(self, user_states: torch.Tensor) -> torch.Tensor:
+        """Reflection-aware scoring: refine user states with contrastive signal."""
+        # Reflect user state through a lightweight MLP for nuanced features
+        h = self.output_norm(
+            self.input_dropout(
+                torch.relu(self.reflection_fc1(user_states))
+            )
+        )
+        catalog = self.item_embedding.weight[1:]
+        ref_logits = h @ catalog.t() + self.item_bias[1:]
+        return ref_logits
+
+    def forward(self, histories: torch.Tensor) -> torch.Tensor:
+        """Override to enable reflection-based scoring."""
+        user_states = self.encode(histories)
+        base_logits = self.catalog_logits(user_states)
+        ref_logits = self.reflection_logits(user_states)
+        # Learnable gate to blend base and reflection scores
+        gate = torch.sigmoid(self.gate_weight)
+        mixed_logits = (1.0 - gate) * base_logits + gate * ref_logits
+        return mixed_logits

```

产物：[训练记录](../../.reclab/sc/sessions/20261120T020207Z-compare-eeddcd55/strategies/reflection/experiments/exp-9cb5ed2900e269883155dc7070b9cb55/train/training.json) · [训练日志](../../.reclab/sc/sessions/20261120T020207Z-compare-eeddcd55/strategies/reflection/experiments/exp-9cb5ed2900e269883155dc7070b9cb55/train-log.json) · [模型权重](../../.reclab/sc/sessions/20261120T020207Z-compare-eeddcd55/strategies/reflection/experiments/exp-9cb5ed2900e269883155dc7070b9cb55/train/best.pt) · [评分证据](../../.reclab/sc/sessions/20261120T020207Z-compare-eeddcd55/strategies/reflection/experiments/exp-9cb5ed2900e269883155dc7070b9cb55/grade.json) · [源码 · configs/default.json](../../.reclab/sc/sessions/20261120T020207Z-compare-eeddcd55/strategies/reflection/snapshots/0e66288a361fce2f71ea7388110b72813474fc277993cfb976c6f3d7ea4f1ca7/configs/default.json) · [源码 · src/sasrec.py](../../.reclab/sc/sessions/20261120T020207Z-compare-eeddcd55/strategies/reflection/snapshots/0e66288a361fce2f71ea7388110b72813474fc277993cfb976c6f3d7ea4f1ca7/src/sasrec.py)

### Reflection · candidate-3

src/sasrec.py（冻结源码快照）：

```diff
--- src/sasrec.py
+++ src/sasrec.py
@@ -38,9 +38,13 @@
         self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
         self.output_norm = nn.LayerNorm(hidden_size)
         self.item_bias = nn.Parameter(torch.zeros(num_items + 1))
-        # Reflection enhancement components
-        self.reflection_fc1 = nn.Linear(hidden_size, hidden_size)
-        self.gate_weight = nn.Parameter(torch.zeros(1))
+        # Adaptive reflection enhancement
+        self.reflection_mlp = nn.Sequential(
+            nn.Linear(hidden_size, hidden_size),
+            nn.ReLU(),
+            nn.Linear(hidden_size, hidden_size),
+        )
+        self.adaptive_gate = nn.Linear(hidden_size, 1)
         self.reset_parameters()

     def reset_parameters(self) -> None:
@@ -70,23 +74,19 @@
         return user_states @ catalog.t() + self.item_bias[1:]

     def reflection_logits(self, user_states: torch.Tensor) -> torch.Tensor:
-        """Reflection-aware scoring: refine user states with contrastive signal."""
-        # Reflect user state through a lightweight MLP for nuanced features
-        h = self.output_norm(
-            self.input_dropout(
-                torch.relu(self.reflection_fc1(user_states))
-            )
-        )
+        """Reflect user state through adaptive MLP to capture latent preferences."""
+        h = self.output_norm(self.input_dropout(self.reflection_mlp(user_states)))
         catalog = self.item_embedding.weight[1:]
-        ref_logits = h @ catalog.t() + self.item_bias[1:]
-        return ref_logits
+        return h @ catalog.t() + self.item_bias[1:]

     def forward(self, histories: torch.Tensor) -> torch.Tensor:
-        """Override to enable reflection-based scoring."""
+        """Enable sequence-aware reflection-based scoring."""
         user_states = self.encode(histories)
         base_logits = self.catalog_logits(user_states)
         ref_logits = self.reflection_logits(user_states)
-        # Learnable gate to blend base and reflection scores
-        gate = torch.sigmoid(self.gate_weight)
-        mixed_logits = (1.0 - gate) * base_logits + gate * ref_logits
+
+        # Adaptive gating: condition blend ratio on state representation
+        gate_logit = self.adaptive_gate(user_states).squeeze(-1)
+        gate = torch.sigmoid(gate_logit)
+        mixed_logits = (1.0 - gate.unsqueeze(1)) * base_logits + gate.unsqueeze(1) * ref_logits
         return mixed_logits

```

产物：[训练记录](../../.reclab/sc/sessions/20261120T020207Z-compare-eeddcd55/strategies/reflection/experiments/exp-747a184da0f57a9af7c28f92ada87b8d/train/training.json) · [训练日志](../../.reclab/sc/sessions/20261120T020207Z-compare-eeddcd55/strategies/reflection/experiments/exp-747a184da0f57a9af7c28f92ada87b8d/train-log.json) · [评分证据](../../.reclab/sc/sessions/20261120T020207Z-compare-eeddcd55/strategies/reflection/experiments/exp-747a184da0f57a9af7c28f92ada87b8d/grade.json) · [源码 · configs/default.json](../../.reclab/sc/sessions/20261120T020207Z-compare-eeddcd55/strategies/reflection/snapshots/576a677a723f6264f95564907c9b46c7f43d1206b968ce736ec31fc918d4ace7/configs/default.json) · [源码 · src/sasrec.py](../../.reclab/sc/sessions/20261120T020207Z-compare-eeddcd55/strategies/reflection/snapshots/576a677a723f6264f95564907c9b46c7f43d1206b968ce736ec31fc918d4ace7/src/sasrec.py)

### Reflection · candidate-4

src/sasrec.py（冻结源码快照）：

```diff
--- src/sasrec.py
+++ src/sasrec.py
@@ -38,8 +38,13 @@
         self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
         self.output_norm = nn.LayerNorm(hidden_size)
         self.item_bias = nn.Parameter(torch.zeros(num_items + 1))
-        # Reflection enhancement components
-        self.reflection_fc1 = nn.Linear(hidden_size, hidden_size)
+        # Refined reflection enhancement with deeper MLP and stability
+        self.reflection_mlp = nn.Sequential(
+            nn.Linear(hidden_size, hidden_size),
+            nn.ReLU(),
+            nn.LayerNorm(hidden_size),
+            nn.Linear(hidden_size, hidden_size)
+        )
         self.gate_weight = nn.Parameter(torch.zeros(1))
         self.reset_parameters()


```

产物：[训练记录](../../.reclab/sc/sessions/20261120T020207Z-compare-eeddcd55/strategies/reflection/experiments/exp-7bad925abfa75d3e80486fb05756c117/train/training.json) · [训练日志](../../.reclab/sc/sessions/20261120T020207Z-compare-eeddcd55/strategies/reflection/experiments/exp-7bad925abfa75d3e80486fb05756c117/train-log.json) · [评分证据](../../.reclab/sc/sessions/20261120T020207Z-compare-eeddcd55/strategies/reflection/experiments/exp-7bad925abfa75d3e80486fb05756c117/grade.json) · [源码 · configs/default.json](../../.reclab/sc/sessions/20261120T020207Z-compare-eeddcd55/strategies/reflection/snapshots/79657896ab452d9f8e26e2c146fbd7815e0c5ff152d2eabb34cfc9b03f97f085/configs/default.json) · [源码 · src/sasrec.py](../../.reclab/sc/sessions/20261120T020207Z-compare-eeddcd55/strategies/reflection/snapshots/79657896ab452d9f8e26e2c146fbd7815e0c5ff152d2eabb34cfc9b03f97f085/src/sasrec.py)

## 补充审计与诊断

- Reflection candidate-1 的 catalog\_logits 与 reflection\_logits 互相调用，触发 RecursionError；此项与其余候选的非有限 logits 故障不同。
- Reflection candidate-2 新增混合 forward 分支，但固定 train/runtime 实际调用 catalog\_logits\(encode\(...\)\)，该分支未参与训练和预测。新增 Linear 初始化会消耗随机数，不能把分数变化归因于未调用的 MLP 分支，也不能假定原路径初始权重与基线相同。
- Reflection candidate-3 的两层 MLP 和 adaptive\_gate 同样仅在未调用的 forward 中使用；配置、train 和 runtime 与 candidate-2 相同。不能把其分数变化归因于新分支。
- Reflection candidate-4 的新增 MLP 和 gate 仍未接入训练/预测路径；reflection\_logits 还引用已删除的 reflection\_fc1，若调用将触发 AttributeError。该潜在缺陷未在实际路径触发。critic 审核通过不构成新结构生效或因果归因正确的证据。
- 32 个微型条件中 30 个输出有限、2 个出现非有限 logits，没有执行训练或载入 checkpoint，初始化参数均有限，推理前后参数与缓冲哈希未改变。
- 两处 NaN 都是 hidden64、两层、四头、左填充且允许 MHA fastpath 的条件（norm\_first True/False 各一个）；同模型、同输入关闭 fastpath 后输出有限。
- 该探针表明至少存在无需训练即可触发的推理数值问题，不支持将主实验失败一概归因于梯度爆炸。探针使用微型随机初始化输入，不能独自证明所有真实失败的完整根因，也没有记录具体执行内核。

## 结论边界与下一步

- 由独立评分器对同一验证集的预测评分；验证集也用于 checkpoint 选择和候选搜索，分数提升不等于最终测试集提升。
- 本轮未评测最终测试集。
- 仅使用一个训练种子，不能据此宣称统计显著提升或某种 Agent 模式普遍更优。
- 未核验实际货币账单；只有记录中的调用量和作业耗时可用于本轮资源对照。

保留本轮已评分的最佳源码与模型产物，再用额外训练种子和最终测试集验证改进是否稳定。

[机器可读实验记录](../strategy-comparison-20261120.json)
