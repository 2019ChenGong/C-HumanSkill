# ELEMK — 分解式池化管线 v2 设计稿(预注册,先导 = MAD k8_s0)

> ### ⚠️ 全局修正 · 2026-08-06 · 丢行准入线 5% → 10%
>
> 本文件全篇约 20 处把「丢行率 ≤5%」写成建卡门/kill 线。**该线自 2026-08-06 改为 10%**,
> 现行规则 = [`DROPGATE_DECISION.md`](DROPGATE_DECISION.md)(阈值 + 纪律 D1–D4)。
>
> - **这是裁定,不是推导。** 5% 本身也从无推导 —— 它是干勾期丢行一度 28.9% 之后,
>   设在实测 ~3% 之上的一条哨兵线(`V6_METHOD_AND_DATA.md` L64)。论文中**禁止**写成
>   「阈值经实验确定」。
> - **本文所有历史判决登记(R1 / R6 / R6d / R6e / R6f / R9 / R13 …)原样保留,一个字不改** ——
>   它们如实记录了当时按 5% 门作出的判决。**下表列出哪些判决因此被重新打开:**
>
> | 历史判决 | 丢行 | 10% 门下 |
> |---|---|---|
> | qwen3.7-plus 重建(L892 区) | 8.7% | **复活**,须 vs `deepseek@8.7%` 匹配对照重测 |
> | GLM-5.1 重建(L925) | 9.4% | **复活**,须 vs `deepseek@9.4%` 匹配对照重测 |
> | qwen3.7-max(L961) | 11.2% | D3 的 10–25% 带:可测,须匹配 + 披露 |
> | kimi-k2.6+GUIDE(L1052) | 14.7% | 同上 |
> | MiniMax-m2.7(L926) | 69.0% | **仍作废**(>25%,消融匹配失效) |
> | CV k8_s2 卡集(L463) | 7.4% | **复活**,侧车尚在 ⇒ 免费;R1「CV 未认证」须含 s2 重跑 |
> | deepseek+GUIDE 对照(L1010) | 4.79% | 原本就过,不变 |
>
> **⇒ R6「deepseek 是目前唯一裸插同时过三难的改写器」即刻降级为「待定」**,
> 在 qwen-plus / glm-5.1 补完效用与匿名两轴之前不得使用。
> **⇒ `DROPGATE_THRESHOLD_PREREG.md` §5 的回溯义务照常执行 —— 换阈值不豁免回溯。**

2026-07-14(#117)。把黑箱一句 prompt 的池化拆成显式管线:**抽取 → 对齐 → 数支持度 → ≥Q 过滤 → 成文**。
本文件在跑任何实验**之前**冻结全部阈值与 prompt(防 p-hacking)。用户拍板:上;Q=2 和 Q=3 都跑;灰区仲裁 = 免费 sonnet-4.6 子代理。

## 动机(实证,来自黑箱解剖 2026-07-14)
现行 neutral_fixed 黑箱卡在**要点级是拼贴不是共识**:35.4% 要点是单成员近逐字复制(cos≥0.9,逐卡 10–59%,
**在退化修复过的卡上**——卡级修复管不到要点级);每卡 37–69% 内容来自 top-2 成员;成员平均仅 21% 要点入卡。
prompt 说"只留共有",LLM 实际做"挑几个人的句子改写"。→ 把"共有"变成可数、可审计、带保证的操作。
查重:无人做过"抽取→对齐→支持度→阈值→成文"用于 k 人文本隐私(最近邻 = Belief-Level Aggregation
arXiv 2601.04889,同管线形状、目的是忠实性非隐私;Clio/Urania 是百万级分析、非 k=8 可用工件;隐私鼻祖=PATE/DPSU/k^m-anonymity)。

## 管线(冻结)
| 步 | 做法 | 冻结参数 |
|---|---|---|
| ① 抽取 | deepseek 把每张成员 `aggro` 卡拆成**自足原子决策要点**(禁模板/标题/脚手架;保留阈值与条件;8–30 词/条;JSON 数组) | temp 0.2, max_tokens 2000;缓存按成员 |
| ② 对齐 | 每簇内:要点 × 其他成员,best-match 余弦(text-embedding-3-small)。**≥0.65 自动同;<0.45 自动异;[0.45,0.65) 灰区 → sonnet-4.6 子代理二判**("同一条可执行决策规则吗?"保守:仅当同规则才判同——过度合并=虚增支持度=隐私 bug) | AUTO_YES=0.65 / AUTO_NO=0.45;批 ≤50 项;波次 ≤8 |
| ③ 支持度 | support(e) = 1 + #{其他成员有匹配要点}(自动同 ∪ 灰区判同) | 纯计算 |
| ④ 过滤 | 留 support ≥ Q;语义去重 cos≥0.80 留最高支持度代表。**Q=2 与 Q=3 两臂** | 输出支持度直方图+被砍清单(审计工件) |
| ⑤ 成文 | conspf 的 ASSEMBLE prompt(只用共识要点、不得新增/回渗),deepseek 1300 tok;**反漂移后检**沿用 conspf(成文行 support==1@TAU=0.55 且 top-cos≥HI=0.75 → 删) | 与 conspf 仪器一致,可比 |

产物:`data/20mad/cmd_shared_cards_mad__elemk_q2.json` / `__elemk_q3.json` + `_stats` sidecar(逐簇 n_elements/kept/支持度直方图/leak_before/after/words)。
脚本:`scripts/elemk_build.py`(STAGE=cost|extract|pairs|fuse)。要点缓存 `data/20mad/elemk_elements_k8_s0.json`;灰区批 `results/mad/elemk_adj/`。

## 相对旧 conspf 的三处修复
1. **按行抽取→LLM 原子抽取**(旧版继承蒸馏模板骨架,模板句制造假支持度;文献:粒度不固定则支持度不可复现);
2. **单阈值嵌入→三段式**(自动同/自动异/灰区仲裁;旧版 TAU=0.55 一刀切,真同义落 .45–.65 灰区误判);
3. **评估搬到不饱和处**(旧版在 2AFC k8 饱和轴上看≡无操作;本次直接测要点级可追溯 + FC 效用,2AFC k8 只做不变差)。

## 评估闸(顺序=风险优先)
- **G0 建卡**:支持度直方图、kept%、thin 卡告警(kept<3 或 <60 词)。
- **G1 要点级(直接靶点,$0 本地)**:单成员可追溯率(黑箱 35.4% → elemk 预期 ~0 by construction)+ 逐字 census 对比 neutral_fixed。
- **G2 FC 效用(最高风险,先测)**:`mad_fc_export` 复用 `nec` env 门(`NEUTRALCLEAN`→elemk 卡,`CONTRASTS=ne-nec`),先 30-dev(240 units)/Q 臂;判官免费 sonnet 按 SKILL 协议。**过闸标准:elemk 对 ne 的 TOST δ=.10 平局或更好**;掉效用 SIG → 该 Q 臂死(旧 conspf 曾 −0.040,但那是废弃仪器)。
- **G3 2AFC k8 不变差($0)**:`neutral_2afc_export` NEUTRALC→elemk 卡,单 pack/Q 臂,与 neutral_fixed .573 同尺对比。
- 对照臂:random-drop 同数量(证明是共识不是删得多)——G1 层面即可($0,复用 degen_kanon 模式)。

## 价值主张(诚实,预登记)
预期赢的:方法 make sense(可数/可审计/元素级 k-匿名保证)+ 结构性根治要点级拼贴(35%→~0)+ 动机发现。
**不承诺**:k8 2AFC 头条变好(轴已饱和,旧 conspf 即 .496)。若 G2 效用掉 → 如实报,该臂死。

## 预算(count_tokens 实测后填)
估:抽取 128 call ≈ $0.25 · 嵌入 ≈ $0.03 · 成文 32 call ≈ $0.07 · G2 起草 480 drafts ≈ $0.5 → 先导 ≈ **$0.9**;
G2 若过闸扩全量 128 dev(2 臂 2048 drafts)≈ 再 +$2。仲裁/判官/攻击器全免费 sonnet。

---

# V3 修复臂附录(2026-07-14,#119,跑前冻结)

依据 V2b 归因(`results/mad/ELEMK_WHY_LOSES_FINDINGS.md`):elemk 掉效用的近因 = 共识要点以"无条件门禁政策"
的形态成文,起草模型照办误开火(仅-nec-punt 单元黑箱赢 .864/.865 = 约一半差距)。
**V3 = 最小因果对**:管线、要点、支持度过滤、去重、反漂移后检**全部不动**,只改成文步的"渲染方式":
1. **卡首固定前言**(模板文字,非成员内容):"Background reference ... engage the specific case first,
   apply a rule only where its condition actually holds."(卡是背景参考,不是执行程序)
2. **门禁强制降格为最后手段条件式**:所有"索要信息/repro、扣住不动、关 incomplete"类动作必须写成
   "先检查报告里已有什么,仅当确实缺失才……",禁止写成无条件命令或 Step-1。
3. 其余成文指令与 V2 逐字一致(只用共识要点、不得新增、具体可用)。
   (刻意**不做**"门禁占比上限"——那是内容改动,会破坏最小因果对。)

实现:`elemk_build.py` env `ASM=v3`(默认 v2 不变),输出 `cmd_shared_cards_mad__elemk_q2v3.json`。
仅跑 **Q=2**(离平局最近的臂)。G2 包:同 V2 配置(SEED=0, NEXPERT=30, CONTRASTS=ne-nec,
PROBE_ARM=ne, DELTA=.10),BATCHDIR=`results/mad/fc_elemk_q2v3`;ne 起草命中 chat 缓存,
nec 起草因卡文本变化自动新键(无脏缓存)。判官 = 免费 sonnet 子代理(显式 model: sonnet),
波次 ≤8,覆盖校验 + 补批模式。

**预注册成功标准(三条,跑前冻结;sonnet 评审 #8 修订于出结果前):**
- **主指标**:ne−nec(v3) 达到 **TIE 认证**(δ=.10,电池过 + sMDE<δ + CI⊂±δ)才算赢。
  仍 DIFFERENT → V3 死;落在 UNDERPOWERED / inconclusive / battery-fail → **非胜利**,按其本名如实报。
- **机制指标**:nec(v3) 起草 punt 率(REQ 正则,钉死在 `scripts/elemk_v3_gates.py`,须先在 V2 q2 包上
  复现 ne .542 / nec .583 才有效)从 .583 降到 ≤.56。
- **保证不破**:v3 卡单成员近逐字率(`elemk_v3_gates.py` 同一次调用里对 q2 与 q2v3 同尺计算)
  不高于 q2 + 3pp;卡仍只由 support≥2 要点 + 模板连接文构成。
附加报告义务(评审 #6/#7):v3 包 `draft_len.json` 的 ne/nec 长度差 >V2 的 ~2% 即列为 caveat 讨论;
nec(v3) 起草需查"前言鹦鹉学舌"(engage the specific case / background reference 等片段渗入起草)。
预算:成文 16 call ≈ $0.03 + nec 起草 240 ≈(COST=1 实测 ~$0.12)+ 判官 $0。
**[结果 2026-07-14:V3 失败——ne 仍赢 .565 DIFFERENT,punt 率 .583→.625 反升。归因收窄:根源在共识
材料(交集=门禁),渲染救不了。见 ELEMK_WHY_LOSES_FINDINGS.md 底部。]**

---

# V4 载体–内容分离附录(2026-07-14,#120,跑前冻结)

依据:黑箱卡配方实测(865 词/卡,门禁行 15.0%,启发行 15.4%;q2 卡 475 词、门禁 17.1% 且 85% 无条件、
居标题位)+ V3 判决(材料是根源)→ **把黑箱赢的机制"稀释、不发号施令"重构为显式步骤**。

## 管线(⑤ 步,①②③ 与 elemk 完全不动)
| 步 | 做法 | 隐私性质 |
|---|---|---|
| ①②③ | 抽取 → 三段对齐+仲裁 → support≥2 过滤+去重(= q2 的 kept 集,逐字复用) | 每条 ≥2 人支持,元素级证书 |
| ④ **载体合成** | 冻结 prompt 一次性生成 ~400 词领域顾问文本(bug-triage 的 engagement 优先/给机制假说/权衡考量);**上下文零成员数据;16 簇共用同一份(逐字节相同)**;prompt 明令禁止任何"索要信息/扣住/关闭"类门禁语言 | 结构上零隐私(与成员数据无信息通道) |
| ⑤ **确定性组装** | 载体前 2/3 段落 + "**Shared practices distilled from the team**"节(认证要点**原文**列入)+ 载体后 1/3;**无 LLM 参与** | 成员来源内容与载体逐字节可分;无漂移通道 → **反漂移后检对 v4 不适用**(该检为 LLM 成文的漂移设计;v4 的证书由构造精确成立,守卫改由 G1' 普查承担) |

产物 `data/20mad/cmd_shared_cards_mad__elemk_q2v4.json` + 载体 `data/20mad/elemk_carrier_mad.txt`(入库审计)。
配方目标:总量 ~850-900 词,门禁行占比 ≤ ne 的 15%,门禁不得居标题/首步位。

## 预注册闸(顺序执行,前闸不过不花后闸的钱/工)
1. **G1' 普查($0)**:`elemk_v3_gates.py MODE=census` 同轮计算 q2/q2v3/q2v4,v4 单成员近逐字率 ≤ q2+3pp。
   **[修正 2026-07-14,起草前]**:首次普查 5.2% > 3pp,闸判失败——v4 原文引用要点,其中支持度来自灰区
   仲裁的要点(内容 ≥2 人、措辞嵌入仅贴 1 人)构成措辞级暴露。修复 = 新增 **⑤' 措辞匿名过滤**
   (conspf 后检同款 TAU=.55/HI=.75,前移至组装前:措辞层 n55==1 且 top≥HI 的要点不进卡,
   `wording_dropped` 记入 stats sidecar 审计)。修复后重过本闸方可起草。
2. **punt 预闸($0.12 起草后、判官前)**:`MODE=punt` 于 v4 包,**nec punt 率 ≤ .56** 才派判官;
   >.56 → V4 机制通道未修复,判官波不派,如实报(省下 22 批判官工作量)。
3. **主指标**:ne−nec(v4) **TIE 认证**(δ=.10,电池过+sMDE<δ+CI⊂±δ)= 赢;
   DIFFERENT → V4 死;UNDERPOWERED/inconclusive/battery-fail → 非胜利,按本名报。
附加报告:draft_len ne/nec 差 >2% 列 caveat;载体片段鹦鹉学舌检查(PARROT 正则扩充载体特征短语)。
**[评审修正 2026-07-14,起草前]** sonnet 评审 BLOCKER-1:载体 prompt 与判官 rubric 存在短语级重叠
("mechanism hypothesis"/"duplicates…trade-offs"),单独的 TIE 无法归因(ne 卡自身也含同款内容,
故重叠对两臂对称,但审稿人无法事后区分)→ **纯载体对照臂升级为本轮必跑**(nec=纯载体卡,16 键同一文本,
同 240 units)。**预注册解释矩阵(跑前冻结):**
| v4 结果 | 纯载体结果 | 唯一解读 |
|---|---|---|
| TIE | TIE | MAD 上卡内容惰性(与 in≈sham 一致);v4 与黑箱效用等价,**增益全在证书轴**;不得宣称"共识内容有效用" |
| TIE | ne 赢 SIG | **认证共识内容有正边际效用** —— 最强结果,Pareto ≥ 黑箱成立且内容非虚设 |
| ne 赢 SIG | 任意 | V4 死,如实报(第四次"纯度换效用") |
证书措辞澄清(评审 #5,写进最终报告):support≥2 认证的是**规则为 ≥2 人所持**;留存文字的表面措辞
是最高支持度成员的原话(经 ⑤' 措辞匿名过滤,其措辞嵌入不再唯一贴单一成员)——不是多人联合撰写的句子。
punt 预闸附加义务(评审 #3):REQ 通过后人工抽读 ≥15 条未命中 REQ 的 nec 起草,确认无"同义词 punt"。
预算(含对照臂):载体 ~$0.01 + 起草 ~$0.12×2 + 判官/评审 $0 ≈ **$0.25**。

# V4-X 外推附录(2026-07-14,#122,跑前冻结):V4 上 CV + Enron

**科学问题**:MAD 结论"成员内容浓度∝效用损失"是 MAD 特异(`in≈sham`,卡内容≈惰性)还是普适?
CV/Enron 上 `in>sham` SIG(卡内容有域价值)⇒ 这是载体–内容配比的**另一个 regime**,不是同一实验的重复。

## 预注册预测(跑前冻结,每数据集独立判)
- **P1(载体单独不够)**:`ne−carrier` 不再显著 <.5(载体不再赢黑箱);更强形式 = ne 赢 SIG。
  依据:`in>sham` ⇒ 卡的域内容有真实边际,零成员载体没有它。若载体仍显著赢 ⇒ "in>sham=内容有价值"
  的解读要重审(或 rubric 伪影),如实报。
- **P2(共识块扛回域价值)**:v4 − 纯载体的边际在这两家翻正(v4 优于载体方向)。
- **P3(headline,判决标准)**:`ne−v4` TIE 认证或不差(UNDERPOWERED 且点估计 ≤.5 = 方向达标非胜利);
  **DIFFERENT-输 ⇒ V4 在该数据集不外推,如实报**。P1/P2 是机制解释,P3 独立定生死。
- 跨数据集最强可写形式(若 P1 翻转 + P3 不差):"载体/内容最优配比是数据集依赖的,但 V4 管线在
  内容惰性(MAD)与内容有价(CV/Enron)两种 regime 下都追平黑箱"。

## 管线(与 MAD V4 逐步同;仅源数据与载体域不同)
| 项 | Enron | CV |
|---|---|---|
| 分区 | K=8 **SEED=1**(正则分区,匹配已发运 neutral_fixed/concat/FC 仪器) | K=8 SEED=0 |
| 簇数 | 14 | 9 |
| 抽取源 | `aggro`(独立已通用化卡) | `aggro`=`nuwa`(CV 无独立 aggro,**登记为已知差异**;support≥2+⑤′ 照常适用) |
| 卡文件 | `cmd_shared_cards__elemk_q2v4.json` | `cmd_shared_cards_cv__elemk_q2v4.json` |
| 载体 | `elemk_carrier_enron.txt` | `elemk_carrier_cv.txt` |
| 纯载体卡 | `cmd_shared_cards__carrier_only.json` | `cmd_shared_cards_cv__carrier_only.json` |

阈值全部冻结同 MAD:AUTO_YES=.65 / AUTO_NO=.45 / **Q=2** / DEDUP=.80 / ⑤′ TAU=.55 / HI=.75;
灰区仲裁 = 显式 `model: sonnet` 子代理,批 ≤50;组装 = compose_v4 确定性(无 LLM);
载体一次生成、该数据集全部簇逐字节共用、prompt 禁门禁语言。

## 载体 prompt(冻结;避开该数据集判官 rubric 关键词——Enron 避 sound/risks/trade-offs/actionable,
## CV 避 correct/caveats/conditions;残余重叠登记 caveat,ne−v4 与 v4−carrier 差值对其免疫)
- **Enron**:~400 词,工作邮件情境(批准/请求/日程/协调/外部伙伴);(1) 先读懂这封邮件本身——发件人
  实际在要什么、每种回应方式对相关人和已有承诺的后果;(2) 批准 vs 婉拒、承诺 vs 留余地、抄送同事、
  回复时机 = 可权衡的可能性,非流程;(3) 全部措辞为考量而非命令;(4) 禁"索要更多信息/暂缓回复/
  默认上报"类指令;(5) 短小 markdown 节+简短 bullet,无编号强制步骤。
- **CV**:~400 词,应用统计答疑;(1) 先读懂这道题——数据/模型/目标已给出什么,提问者背后的实际
  问题是什么,再谈方法;(2) 模型假设、样本量、测量质量、多重检验暴露、简单 vs 复杂方法的取舍 =
  可权衡的可能性,非流程;(3) 考量措辞;(4) 禁"索要澄清/暂缓作答/拒答"类指令;(5) 同上。

## 预注册闸(顺序执行;每数据集独立走完)
1. **G0 建卡统计**:支持度直方图 / kept% / thin 告警(kept<3 或 <60 词)入 stats sidecar。
2. **G1′ 普查($0,起草前)**:同轮计算 {ne=neutral_fixed, q2v4}(通用化 census,卡行 vs 成员元素,
   cos≥.9 单成员近逐字):v4 目标 0.0%,硬上限 = 同数据集 ne 的实测率(same-run 对照);载体 6-词
   shingle 渗漏普查。fail ⇒ 修复重过才准起草(MAD 先例:⑤′ 就是这么来的)。
3. **G2 效用(kill gate)**:每数据集两包 —— `CONTRASTS=ne-nec` + `NEUTRALCLEAN`={q2v4 卡, carrier_only 卡},
   `PROBE_ARM=ne`;判官 = 显式 sonnet 子代理波 ≤8;电池必须 PASS(fail ⇒ 站规 ≥2 判官上 r2);
   判定词典 = cv_fc_score DELTA=.10。**功效预算(诚实预登记)**:Enron ne−nec 按簇去重 = 14 簇 ×8 任务
   ×2 序 = 224 项/包(与 ne−cc .607 SIG 同功效);CV = 231 单元 ×2 序/包但 9 簇聚类 —— ne−* 已知坐
   9 簇墙,**UNDERPOWERED 风险预先接受,按词典本名报**。
4. **诊断(非闸,不判生死)**:punt 率(REQ 正则;MAD 基线在此数据集无复现锚 ⇒ 只报 ne/nec 对比与
   方向,不设阈值);draft_len 差 >2% 列 caveat;PARROT/载体 shingle 检查。
5. **G3 匿名($0,G2 过了才跑)**:2AFC 不变差 vs 同数据集 neutral_fixed(同包三通道 indiv/neutral/conspf←v4,
   MAD #121 模板;Enron 用 k8_s1)。

## 预算与产物
建卡(抽取 Enron 116 + CV 77 张 aggro 卡,deepseek)+ 仲裁($0 sonnet)+ nec 起草(Enron 每包
14 簇×8 任务=112 drafts、CV 每包 231;各 ×2 包)——精确数额以 STAGE=cost 实报为准,预估 ≈$1 内两家全包。
判决文档 `results/{enron,se}/…V4X…FINDINGS`(合并一份放 `results/V4X_CROSSDATASET_FINDINGS.md`)。
代码改动(与本附录同评审):`elemk_build.py` 数据集通用化(ADJ/卡名/载体名/载体 prompt 按 DS)+
`STAGE=carrieronly`;`cv_fc_export.py`/`enron_fc_export.py` 移植 mad 版 `CONTRASTS`/`NEUTRALCLEAN`(nec 臂)/
`PROBE_ARM` 三 env(Enron 的 nec 入 PERCLUSTER=按簇起草+去重;CV 的 nec 入 config.pooled_arms=按簇 bootstrap);
`elemk_v3_gates.py` 通用化(rubric 标签/载体路径/census 文件表按 DS)。

**[V4-X 评审修正 2026-07-14,跑前落实]** sonnet 对抗评审(PROCEED-WITH-FIXES):
- BLOCKER-1 已修:仲裁目录按 (K,SEED) 命名空间(`elemk_adj_k{K}_s{SEED}`,MAD 留旧路径)+ `stage_fuse`
  对 pairs.json 簇键与本次分区断言一致(漏设 SEED=1 时支持度静默塌 1 的通道封死)。
- MAJOR-2 已修:census 断言各数据集正则 (K,SEED)(mad 8,0 / enron 8,1 / cv 8,0)。
- MAJOR-3 已修:载体生成后对该数据集 rubric 关键词做 $0 字面扫描;命中则附加显式禁词重试一次,
  两次扫描结果都入报告(MAD 先例:prompt 避词不可靠,载体仍回声 "duplicates"/"condition")。
- MAJOR-4 登记 caveat:CV(带参照判官)vs Enron(无参照判官)的 rubric 类型与数据集身份混淆——
  P1 的跨数据集比较若两家分歧,不能全归"内容价值 regime",写结论时必须带此 caveat。
- MAJOR-5 登记义务:CV 抽取源=原始个体卡(aggro==nuwa),冻结阈值是在 MAD 的已通用化 aggro 上标定的
  —— 报 CV 的 auto-yes/no/gray 三段占比 + ⑤′ 措辞过滤丢弃率,与 MAD/Enron 并排目检后才信 G1'/kept%。
- MAJOR-6 有意识接受:punt 预闸不适用于新数据集(无标定锚),机制性失败只能在判官波之后发现;
  ~$1 总预算下接受此风险。
- MINOR:跑 QS=2;cv_fc_score 补 ("ne","nec") 标签(已落);MAD 回归 = fuse 重跑字节对比(跑前执行)。

---

# V5 附录:消毒不重建(sanitize, don't rebuild)—— 三家统一管线(2026-07-14,#123,预注册)

## 动机(用户指令 + $0 解剖)
用户:模拟贴近黑箱卡的生成方式,综合一套 3 数据集都好用的流程。V4-X 判决:重建路线(载体+共识)在
in>sham 数据集显著输黑箱,损失 ∝ 被删的成员内容。$0 解剖(`elemk_v3_gates.py MODE=anatomy`,三家同轮):

| | MAD | CV | Enron |
|---|---|---|---|
| ne 卡行数/词数 | 661 行 / 865 词 | 260 / 732 | 474 / 860 |
| 行可溯源到成员元素(cos≥.55) | 93.9% | 90.8% | 92.0% |
| 归因行(≥.65)中 sup=1 占比 | 60.5% | 61.3% | **86.0%** |
| 行与成员 aggro 卡共享 6 词连串 | **80.0%** | **86.9%** | **90.9%** |
| 每卡贡献成员数(@≥.65) | 7.4/8 | 6.6/8 | 6.4/8 |

黑箱配方三家一致 = **广采博收**(主力是单支持内容,几乎全员贡献),不是共识;词面暴露巨大。
⇒ V5 不再替换黑箱的选材/组织(那是效用来源),而是在其上加一层**确定性结构的措辞消毒 + 逐行溯源审计**。
这同时是 V4-X 机制结论的判别实验:效用骑在成员内容的**语义**上(V5 改写后应保住)还是**措辞**上(应掉)。

## 管线定义(FROZEN;见 scripts/v5_sanitize.py)
每簇(canon:MAD k8_s0 / CV k8_s0 / Enron k8_s1),输入 = `__neutral_fixed` 卡 + 本簇成员 elements/aggro:
1. **逐行拆分**:内容行 = ≥5 内容词且非全大写(census 同款 `_card_lines` 正则);其余(标题/短行)原样保留。
2. **逐行改写**(deepseek-chat,T=0.3,冻结 prompt):
   "Rewrite this line from a team skill card in completely fresh wording. Preserve the full actionable
   content — every condition, threshold, quantity, named method or tool, and sequence — but do NOT reuse
   any distinctive phrase of 4+ consecutive words from the original. If the line contains a person's name
   or an organization-internal project name, replace it with a generic role or descriptor. Keep roughly
   the same length and the same bullet/sentence form. Output ONLY the rewritten line."
3. **逐行双闸(确定性,build 内强制)**:
   - **词面闸**:改写行与 {原行 ∪ 本簇全部成员 elements ∪ 本簇全部成员 aggro 卡} 无任何共享 6 词连串
     (`_shingles` n=6,与载体渗漏仪器同一定义);
   - **保真闸**:cos(原行, 改写行) ≥ 0.75(text-embedding-3-small;防改写漂移丢内容)。
   违闸 → 带具体违规反馈重试(≤2 次)→ 仍违 → **丢行并计数**(丢行率入 stats,>5% 即 G-lex FAIL 级异常)。
4. **审计 sidecar**(`__v5san_audit.json`):每行 {原行号, top 元素 (ck,ei), cos, support, 成员数@.55,
   unattributed 标志, 重试数, 是否丢弃}。卡 = `__v5san.json`(行序/标题/结构与 ne 卡逐行对应,无组装 LLM)。

## 证书语义(诚实重定义 —— 与 V4 不同,预先声明)
- **词面证书(新主张)**:v5 卡 0 行与成员文本共享 6 词连串(ne 基线 80–91%)——杀 string-match/引文链接。
- **语义 census(G1′)**:v5 预期 ≈ ne 水平(内容按设计保留)——**如实报,不是主张**;v4 的元素级 0.0% 证书
  v5 不继承,换来的是效用。
- **行为匿名(G3 2AFC)**:本轮**必跑**(v5 保留成员语义,不能借 v4 的 G3);黑箱 ne 带更重词面泄漏仍 ∋.5,
  预测 v5 同样 ∋.5。
- **审计主张**:每行可溯源(元素+support+owner)或显式标 unattributed(LLM 通用,6–9%)。

## 预注册预测与闸(顺序执行,每数据集独立)
- **P1(主科学问题,内容 vs 措辞)**:FC `ne−v5` 三家均**不输**(非 DIFFERENT-lose)。
  kill gate(逐数据集):CI 排除 .5 且 ne 胜 ⇒ 该家判"措辞本身载效用"——注册后果:这是判官措辞偏好
  artifact 的信号,进 caveat 调查而非 V5 直接死(与 V4 的 kill 语义不同,预先声明)。
- **P2(匿名)**:G3 2AFC 三通道(indiv 阳性对照 / ne 同波参照 / v5),v5 CI ∋ .5 三家;配对差 v5−ne 如实报。
- **P3(词面)**:v5 词面 6-shingle 撞成员率 = 0%(build 强制;`MODE=lex` 独立复测),丢行率 ≤5%。
- 诊断(非闸):卡词数 vs ne 差(目标 ±10% 内)、draft_len >2% caveat、punt 率 ne/v5 对比方向。
- G2 FC 配置:`CONTRASTS=ne-nec NEUTRALCLEAN=<v5san 卡> PROBE_ARM=ne`(cv/enron/mad_fc_export 均已支持,
  零代码改动);ne 起草吃 chat 缓存;判官 = 显式 sonnet 子代理波 ≤8,程序化覆盖校验;判定词典 cv_fc_score
  DELTA=.10。功效诚实预登记:CV 9 簇墙 / MAD 16 簇墙 → "认证平局"大概率 UNDERPOWERED,成功判据 =
  不输(非 DIFFERENT-lose),与 P1 一致;MAD 先跑 r1,电池或功效要求时才加 r2。
- G3 包 = #121 模板(conspf 槽注入 v5 卡),三家各一包;免费 sonnet 子代理。
- 顺序:build+双闸+lex($0.2)→ G2 FC 三家(~$0.7)→ P1 不全灭则 G3 三家($0)→ 判决文档。
  预算 ≈ **$1**(改写 ~1400 行 deepseek ~$0.15 + v5 起草 CV 106+Enron 112+MAD 240 ~$0.6 + 杂项)。

## 代码改动(与本附录同评审)
- 新 `scripts/v5_sanitize.py`:STAGE=cost|build,DATASET/K/SEED env,复用 elemk_build 的
  load_clusters/embed/chat/de.pool;输出卡+audit+stats。
- `elemk_v3_gates.py`:已加 MODE=anatomy(本解剖,钉死);加 MODE=lex(任意卡文件词面普查,CARDS env)。
- FC 导出/评分:零改动(env 驱动);cv_fc_score CLAIM ("ne","nec") 标签已在。

**[V5 评审修正 2026-07-14,跑前落实]** sonnet 对抗评审(PROCEED-WITH-FIXES):
- BLOCKER-1 已修:`_PREF` 只剥真弹头(`-`/`•`/`* `/数字./#),**不再剥粗体 `**`**——原版会把
  `**Step 3: …**` 拆成悬空标记送改写(实测三家 ~50% 粗体行受害)= 纯格式混淆;粗体行现在完整送
  LLM,prompt 加"preserve markdown emphasis markers in place";LLM 输出用同一 `_PREF` 剥后重接。
  测量仪器(census/_card_lines)不动(词面 shingle 本来就忽略标点;保持与已发数字可比)。
- BLOCKER-2 已修:STAT_P 逐簇增量写(与卡/audit 同模式),崩溃续跑不再静默丢已完成簇的丢行统计。
- BLOCKER-3 已修(判定收紧,对齐 V4-X 先例):P1 通过 = TIE 认证 或 (UNDERPOWERED/∋.5 **且点估计≤.5**
  = 方向达标非胜利);点估计 ∈(.5,.5+δ) 且 CI∋.5 = "不可判(方向不利)",如实报,计 P1 未证实但非 kill;
  kill = DIFFERENT-lose。kill 后果预注册为具体测试:$0 数值/工具 token 存活审计(build 已内置)——
  存活 ≥95% ⇒ 判"措辞/风格本身载效用(判官偏好 artifact 候选)",V5 在该家死+带 caveat;
  存活 <95% ⇒ 判"改写丢内容",允许**一次**强化保真 prompt 重跑(仅此一次,预注册),再判终局。
- MAJOR-4 已登记:`__v5san_audit.json` = 仅研究内部产物,与原始成员卡同访问等级,**永不与卡共同发布**
  (公开它 = 逐行再识别地图,完全反转匿名)。
- MAJOR-5 已修:build 加确定性数值保全闸——原行数值 token(`\d+(\.\d+)?%?`)必须原样出现在改写行,
  缺失 = 保真类违规重试(带"你丢了数值 X"反馈);大写工具名存活只记 audit 不闸。
- MAJOR-6 已登记(证书范围限定):词面证书基于 `[a-z']` 词序列,**数值/日期 token 不在覆盖范围**
  (它们按设计原样保留 = 内容);论文措辞必须带此 scope。
- MAJOR-7 已预注册:kill 后的判别测试 = BLOCKER-3 的 token 存活审计(风格 vs 内容的 $0 判别);
  风格-only 对照臂(无成员禁词的纯改写)列为可选后手,非默认花费。
- MAJOR-8 已修:FB_LEX 反馈报前 3 个违规 shingle(非 1 个);预注册:某家丢行率 >5% ⇒ 该家停下目检,
  **其余两家照常进行**(不整体停摆)。
- MAJOR-9 已登记:CV aggro==nuwa(原始个体卡,无泛化层)——解剖表/词面表中 CV 的 "vs aggro" 列
  与 MAD/Enron 不同源,跨家比较带此 caveat(继承 V4-X MAJOR-5)。
- MAJOR-10 已修:`_PREF` 字符类与库内规范一致(含 •;见 BLOCKER-1 的新语义)。
- MINOR-11 已登记:V5 MAD r1-only 与 V4 MAD r1+r2 同表比较时必须标注判官轮数/功效差。
- MINOR-12 known:_load_support 三处重复(冻结仪器不动,改一处必须同步三处)。
- MINOR-13 已修:空改写单列反馈(不再误挂"漂移"反馈)。
- MINOR-14 已修:输出文件盖 config 戳(FID/MAXRETRY/SHN/prompt sha1),续跑断言一致,防带病缓存。

**[V5 干跑调参 2026-07-14,G2 前落实(任何效用/匿名结果读取之前)]** MAD G0 单簇干跑两轮:
- 干跑① 丢行 28.9%:一半是 <6 词的粗体短标题(装不下 6 词 shingle = 词面天然安全,却在 FID 闸上
  空转)。修正:改写对象 = ≥6 个 [a-z'] 词的行(SHN 同源规则);更短的行原样保留,词面证书对其
  平凡成立。MAXRETRY 2→4。
- 干跑② 丢行 13.3%,全部 final_fail=fid:为避禁词改写偏狠,余弦落在 [.65,.75)。**丢行=删内容,
  会把 P1 污染成"v5 输在行缺失"**——词面闸不可让步,保真闸改两级:≥FID(.75) 直收;重试耗尽后
  收词面+数值干净且 fid≥FID_FLOOR(.65) 的最佳尝试(audit 标 `fid_tier:relaxed`);再低才真丢。
  FID_FLOOR 入 config 戳。relaxed 行占比入判决文档如实报。

---

# V6 附录:手术式最小改写(minimal-edit sanitize)—— 冲"匿名+效用双追平"(2026-07-15,#124,预注册)

## 动机(V5 三家判决的直接推论)
V5 在 MAD 输 .583 的三通道已定位:①全行 fresh-wording 重写的措辞噪声(双不 punt 单元 ne 仍赢
.632)②punt 率被抬高(.542→.583)③5.1% 丢行。三者皆非隐私必需:词面证书只要求与**成员文本**
无共享 6 词连串;"与原行措辞不同"是 V5 自加的非隐私约束。V6 撤掉它,改为逐行最小干预:
**改写量 ∝ 该行的实际词面暴露**——干净行原样保留,脏行只重写重叠词串。单一机制,逐行自适应,
不需要按数据集分支(分支性由数据自己涌现:MAD 行大多小改,Enron 行改得多)。

## 管线定义(FROZEN;scripts/v5_sanitize.py 新 EDIT=min 分支,输出 __v6min.json)
每簇(canon 同 V5),输入 = neutral_fixed 卡 + 本簇成员 elements∪aggro 词面池:
1. 逐行计算**违规词串** = 该行 6-shingle ∩ 成员池,重构为极大重叠词段(exact runs)。
2. 无违规 → 行**原样保留**(词面证书对其平凡成立)。
3. 有违规 → LLM 最小改写(deepseek T=0.3,冻结 prompt):
   "Minimally edit this line from a team skill card. The ONLY goal: it must no longer contain any of
   the following word sequences (rephrase just enough of each to break the sequence): {runs}. Keep
   every other word of the line VERBATIM — do not paraphrase anything you don't have to. Preserve all
   conditions, thresholds, quantities, named methods, markdown emphasis markers, and the line's
   bullet/sentence form and approximate length. Output ONLY the edited line."
4. **逐行五闸(确定性)**:词面(全行重查:与原行之外的禁池——即成员 elements∪aggro——无共享
   6 词连串;注意 V6 不禁与原行的重叠,这是与 V5 的定义差)/ 数值保全(V5 同款)/ 保真
   cos(原行,新行) ≥ .90(比 V5 紧,因为只动小段;耗尽收 ≥.80 最佳尝试 = relaxed)/
   **punt 状态保持:REQ(原行)==REQ(新行)**(elemk_v3_gates 冻结正则;直接封 V5-MAD 的 punt 通道)/
   行长 ±30%。重试 ≤4 带精确违规反馈。
5. **兜底链(修 V5 的丢行毒药)**:min-edit 耗尽 → 退化为 V5 全行改写(同 V5 闸,≤2 试)→
   仍败才真丢(预期 ≈0;丢行率仍设 5% 异常线)。审计 sidecar 同 V5 + 每行改动词数占比。

## 证书语义(与 V5 的明示差异,预先声明)
- 词面证书:**不变** —— 0 行与成员文本共享 6 词连串(n=6,[a-z'] 词序列,数值 token 例外同 V5)。
- 语义 census:**预期回升到 ne 与 V5 之间**(最小编辑使行对成员元素的余弦基本保留)——如实报,
  注册为"消毒强度刻度盘"叙事:V6-min ↔ V5-full 是同一机制的两档,census 换效用,按合规需求选档。
- 2AFC(G3 必跑,三通道同 V5):黑箱 ne 带 24.3%@.8 近逐字仍 ∋.5,v6 词面严格 ≤ ne ⇒ 预测 ∋.5 ×3
  + 配对差不变差。
- 审计主张不变(sidecar 内部产物,永不随卡发布)。

## 预注册预测与判定
- **P1(效用,主目标"追平")**:FC `ne−v6` 三家均非 DIFFERENT-lose(kill=任一家 DIFFERENT 输);
  进阶目标:**MAD 认证 TIE**(r1 sMDE .097<δ 已可达,需点估计落 [.45,.55] 使 CI⊂±δ);Enron r1 后
  若点 ∈(.45,.55) 且 sMDE≥δ → 加 r2 冲认证;CV 9 簇墙,诚实上限 = 非 DIFFERENT + 点≈.5
  (δ=.10 认证大概率不可达,预先接受)。UNDERPOWERED 方向判定沿用 V5 收紧版(点≤.5 才算方向达标)。
- **P2(匿名)**:2AFC v6 CI ∋ .5 ×3 且配对差 ne−v6 CI 含 0。
- **P3(词面)**:MODE=lex v6 = 0.0% ×3;丢行 ≤5%(预期≈0);census 如实报(预期 ne 与 v5 之间)。
- 诊断(非闸):punt 率 v6 vs ne(应 ≈ 相等,由闸机械保证——若仍差则闸失效要查)、改动词数占比
  分布、draft_len >2% caveat、fallback-to-V5 行占比。
- kill 后果预注册:任一家 DIFFERENT-lose ⇒ 跑同款 token/punt/丢行三判别;若三判别都排除机械原因
  ⇒ 判"该 regime 下词面暴露与效用不可同时保全",V6 收窄为对应 regime 的档位,**不再加第三次
  改写变体**(V5→V6 已是同族第二档,三连改写 = 过拟合判官)。
- G2 配置同 V5(CONTRASTS=ne-nec NEUTRALCLEAN=__v6min PROBE_ARM=ne;MAD SEED=0;判官 sonnet 波 ≤8,
  fc_status 覆盖校验,cv_fc_score DELTA=.10);G3 同 V5(CONSPFC=__v6min)。
- 预算:min-edit 改写(输出比 V5 短)~$0.10 + v6 起草 3 包 ~$0.5 + 可能的 r2(Enron/MAD)~$0.3
  ≈ **$0.6–0.9**;判官/攻击者/评审 sonnet 子代理 $0。

## 代码改动(与本附录同评审)
- `scripts/v5_sanitize.py`:EDIT env("full"=V5 默认,字节不变;"min"=V6)→ 新 prompt/RUNS 反馈/
  五闸/兜底链/输出名 `__v6min{,_audit,_stats}.json`;config 戳加 EDIT 与 punt 正则 sha。
- 仪器零改动(MODE=lex/census/punt/anatomy 全部现成;census/lex 文件表加 v6min 条目)。

**[V6 评审修正 2026-07-15,跑前落实]** sonnet 对抗评审(PROCEED-WITH-FIXES):
- BLOCKER-1 已修:兜底链的 V5 全行改写**补上 punt 状态闸**(违规词串本身就是 punt 短语的行最容易
  掉进兜底——不补则 V6 要封的通道从兜底重开);预注册**兜底使用率上限 5%**(与丢行同线,超线停查)。
- BLOCKER-2 已排:全量建卡前干跑 MAD G0(短行最坏情形)+ Enron G0(词面重叠最高 90.9%),
  目检 tiers 分布 / 兜底率 / punt 类失败计数,再放三家。
- MAJOR-3 已修:保真闸按违规覆盖度缩放——违规词串占行词数 >70%("最小编辑"实质是全行改写)时
  用 V5 阈(.75/.65),否则 .90/.80;预注册,非事后调。
- MAJOR-4 已修:REQ 正则手工复制(elemk_v3_gates import 即执行 MODE,无法直接引)→ sha1 钉死
  (ae0b36dfa10e),漂移即崩。
- MINOR-5 已登记:任何 V5/V6/V4 同表比较必须标注判官轮数与会话(继承 V5 MINOR-11)。
- MINOR-6 已改口径:MAD sMDE .097 是 V5 包的实测值,作为 V6 功效**估计**引用,非保证。
- MINOR-7 已补 else:Enron r1 点估计落在 (.45,.55) 之外且 UNDERPOWERED ⇒ 不加 r2,按词典本名报
  (方向不利=不可判 / 点≤.5=方向达标),不留事后解释空间。
- MINOR-8 known:_hit_runs 给 LLM 的违规词串是小写去标点形态,与原行大小写/标点不逐字对应——
  自愈机制在(每轮全行重查+FB_LEX 报实测违规),代价是可能多一轮重试,计入干跑观察。
- MINOR-9 已核:词面证书 n=6 scope 措辞已在(逐字保留行可能仍含 ≤5 词成员短语,论文措辞必须带
  scope);模块 docstring 的重试预算已修正(≤2→≤4)。

**[V6 干跑修正 2026-07-15,G2 前落实(未读任何效用/匿名结果)]** MAD G0 + Enron G0 干跑:
兜底使用率 50%/37% ≫ 5% 预算,丢行 3.3%/11.1%,3/4 丢行 = punt_drop 双约束死锁。结构清晰:
strict 档改动 6–50%(均值 ~27%,手术目标达成);失败的全是**高覆盖行**——违规词串盖住大半行时
"只改词串、其余逐字"数学上无解(每个 6 词窗都要破)。修正 = MAJOR-3 的完整形态,**按覆盖度路由**:
- cov ≤ 0.70 → min-edit 路线(闸 .90/.80 + punt + 长度,≤5 轮);耗尽 → 转改写路线(与路由行同池
  ≤5 轮)→ 丢。
- cov > 0.70 → **直接走全行改写路线**(V5 prompt + V5 闸 + punt 闸,≤5 轮,relaxed 地板 .65)——
  不再是"兜底",是设计内路由;tier 记 rewrite/rewrite_relaxed(与 min-edit 失败转来的 fallback 区分)。
- 保真覆盖度缩放(原 MAJOR-3 实现)被路由取代;丢行率上限仍 5%(全卡);登记指标:两路线份额、
  各自丢行率、changed_frac 分布(dial 叙事 = changed ∝ 暴露,现在由构造保证)。
**[V6 干跑修正③ 2026-07-15]** 路由后 25/30 行被判高覆盖(与干跑① 14/30 行 min-edit 成功矛盾)
→ 根因 = 覆盖度重复计数:重叠 run 各算一遍。修 `_hit_runs`:重叠/相邻 span 合并为一个 run,
cov = 真并集覆盖。另修 FB_PUNT_DROP:点名 REQ 可用词表('request'/'ask'/'clarify'/'more
information')——正则要字面词,模型语义保留(如 "seek further specifics")正则看不见 = punt_drop
死锁主因。
**[V6 干跑修正④ 2026-07-15,收敛承诺]** 修正③后路由仍吞掉 26/30 行(并集覆盖普遍 >0.7,但干跑①
证明其中一半可被 min-edit 以 changed ~.2-.4 解决)→ 终版配置:路由阈值提到 **cov>0.90**(每个 6 词窗
都被盖住 = 真不可解);0.70<cov≤0.90 的行留在 min-edit 但用 V5 保真阈(.75/.65,MAJOR-3 原案);
其余 .90/.80。**收敛承诺:这是最后一轮建卡调参——干跑只看建卡统计(丢行/份额/changed 分布),
从未看任何效用/匿名数;本轮后无论份额如何,只要丢行 ≤5% 即 GO,不再调。**

---

# R1 附录(任务 #125):V6 多种子匿名 δ=.10 认证(2026-07-15 预注册,评审前冻结稿)

**目标**:"V6 完全替代 CMD" 第一梯队——把 v6 匿名从单种子单波 ∋.5 升级到与黑箱 A1 同级的
3 种子 δ=.10 认证;同批卡供 R2(Enron 效用破簇墙)与 R4(gpt-5.4 包)复用。

## 构建(每家 2 个新种子:MAD s1,s2 / Enron s0,s2 / CV s1,s2)
- 输入 = `*__neutral_fixed.json` 对应种子的 k8 卡(A1 时代已建,退化清零)。
- `v5_sanitize.py EDIT=min`,配置与正典 byte-identical(FID/MINFID/MAXRETRY/SHN/prompt sha 戳
  自动校验);**唯一代码改动:放宽 `_CANON` assert → K=8 且 SEED∈{0,1,2}**。输出增量写入同一
  `__v6min.json`(簇键含种子,无碰撞;配置戳不符即拒续写)。
- 建卡门(同正典):丢行 ≤5%/卡集;词面证书 `MODE=lex` = 0.0%(每新种子独立复测);任一不过 →
  该种子卡集作废排查,不进测量。先 `STAGE=cost` 报价(预算 ~$0.4 改写,超 2× 停下对齐);
  每家先干跑 1 簇(`ONLY=..._G0`)只看建卡统计确认与正典同量级。

## 测量(三通道 2AFC,与正典 v6 波同仪器)
- 每新种子一包:`neutral_2afc_export.py` CHANS=indiv,neutral,conspf(CONSPFC 槽注入 v6 卡),
  nneg=member,env 与正典 v6 波相同除 SEED;目录 `results/{mad,se,enron}/2afc_v6min_s{X}/`。
- 判官 = sonnet 免费子代理(显式 model:sonnet,波 ≤8,禁子代理/禁脚本/逐项独立);覆盖 100%
  (程序化 pid 校验,缺项隔离重判)。
- 波内阳性对照:indiv CI-lo > .5,否则该波作废重跑。

## 合并与预注册判定(δ=.10 主词典,与黑箱 A1 完全一致)
- 池化 = 正典波(2afc_v6min)+ 2 新种子波 = 3 种子;`cmd_multiseed_pool.py` 新增 **MSMODE=v6**
  分支(test chan=`conspf`,参照 chan=`neutral`,门 chan=`indiv`;battery dirs 上述三目录)。
- **P1(主)**:v6 pooled acc,B=(seed,card_id) 与 C=person 双 bootstrap,**认证 = 两者 up95 均 <.60**。
  δ=.05 一并如实报(预期 underpowered,不作主张)。
- **P2**:pooled 配对差 ne−v6(卡聚类):CI 含 0 = 不变差保持;CI 全正只报方向性收益不升级主张。
- **P3**:词面证书 3 种子 × 3 家全 0.0%。
- ne/indiv 通道数字顺带记录为参照与门,不做新主张。
- **kill/诚实分支**:任一 bootstrap up95 ≥.60 → 按本名报 LEAK-not-certified,"完全替代"降级为
  "正典种子不变差、多种子认证未过";先归因(种子方差 vs 攻击者批次方差,照 #64 归因模板),
  **不加第 4 个种子**(黑箱先例:3 种子=正确止损)。不预设 v6 优于 ne;唯一主张=不变差+同级认证。

## 登记的代码改动(仅 2 处)
① `v5_sanitize.py` `_CANON` assert 放宽(见上);② `cmd_multiseed_pool.py` MSMODE=v6 分支。
其余零改动;改后各干跑一遍已有正典数据核对数字复现(byte 级)再动新数据。

**[R1 评审修正 2026-07-15,sonnet 对抗评审 PROCEED-WITH-FIXES,跑前全部落实]**
- **MAJOR-1(入选偏倚)**:正典波是"已观测到 ∋.5 才立项"的预选样本 → 除 3 种子 headline 外,
  **必报仅 2 新种子的次级合并估计**(B/C 同词典),让读者看到认证对预选波的依赖度。
- **MAJOR-2(判定补全)**:P1 认证 = up95<.60 **且 ci95_lo≤.5**(noninf ∧ ¬leak,与
  `cmd_multiseed_pool.certify()` 逐字一致);小而显著的残余泄漏 = LEAK 本名报。
- **MAJOR-3(重跑上限)**:indiv 门失败每波至多重跑 **1 次**;第二次失败 → 走归因分支,不再重掷。
- **MAJOR-4(冻结纪律外延)**:byte 复现核对适用于 R1 期间**任何**代码改动,不限已登记 2 处;
  任何阈值级改动 = R1 预注册作废,须重新注册,不得静默补丁。
- MINOR-5:MSMODE=v6 把 conspf 行重标为 `v6` 再入合并 JSON(防未来按 chan 名误并真 conspf)。
- MINOR-6:neutral 通道排除出认证循环,输出标注 reference-only(防被误读为对 A1 的复认证)。
- MINOR-7:某种子卡集救不活 → 降为 2 种子合并,显式标注 sub-A1-tier。
- MINOR-8:措辞用"达到与 A1 相同的统计认证档(同 δ=.10 词典)",不说"同协议"。
- CLEARED 备案:v6 正典波 sys.txt 与 a1 包 byte 一致(无 guarded 混尺);__v6min.json 多种子增量写
  安全(簇键含种子+配置戳);B/C bootstrap 依赖结构与 A1 逐字同源(分组/配对代码未动)。

**[R1 干跑修正① 2026-07-15]** 构建器依赖 `elemk_elements_k8_s{seed}.json`(查重库 elements 半边),
新种子缺文件。核实:该文件按**成员**键控(成员→元素列表),与分组种子无关;三家正典文件均覆盖
全人口(128/116/77 FULL)→ 新种子文件 = 正典文件逐字节复制(语义恒等,$0;等价于 STAGE=extract
在 llm_cache 全命中下的输出)。属机械依赖补齐,非阈值改动,不触发重注册。
**[R1 干跑修正② 2026-07-15]** 第二依赖:`elemk_adj*/pairs.json`(元素对齐仲裁)只喂审计 sidecar 的
`support` 列,卡内容/路由/闸完全不读;仲裁只建过正典分区。干跑暴露两态皆错:MAD(adj 目录不带种子
后缀)在新种子下静默把 support 全写 1 = 伪造;Enron/CV 直接 FileNotFoundError。修法(审计诚实性,
非阈值):pairs 缺失或分区不匹配 → 该分区所有行 support=null + stats 打 `_support_note` 标;
绝不默认 1。受污染的 mad k8_s1_G0 三文件条目清除重建(缓存命中,$0)。新种子审计的 support 列为
null 是**登记的已知降级**(完整溯源审计可后补仲裁,不阻塞 R1 的匿名认证——认证不读审计)。
**[R1 建卡门判决 2026-07-15]** 丢行率按卡集:MAD s1 3.5%/s2 2.7% ✓,Enron s0 3.1%/s2 2.1% ✓,
CV s1 4.2% ✓,**CV s2 7.4% 超 5% 门 → 卡集作废,不进测量**(预注册条款执行)。排查:失败构成
fid 9/lex 10/punt 1,弥散于 9 簇;典型样本 = s2 两张池化卡把成员模板标题行("# COGNITIVE
OPERATING SYSTEM: Statistical Consulting Protocol")原样搬入 → 与全体成员 aggro 卡撞 6 词库、
改写在 cos≥.75 下无解 → 根因 = s2 批黑箱输入卡的逐字贴近度种子间变异(输入性质,非管线回归;
与"黑箱=拼贴"结论融贯)。**处置(MINOR-7 条款):CV 池化降为 2 种子(正典 s0 + s1),认证结果
显式标注 sub-A1-tier;MAD/Enron 维持 3 种子。**不建 s3(需全套 neutral 合成+退化修复,未注册,
scope 外)。

---

# R3′ 附录(任务 #127,方案修订版):v6 对全部逐人去标识臂的直接 FC(2026-07-15 预注册)

**修订原因(用户)**:替代后 headline = "CMD(=v6) > 逐人去标识";只测 staab 会被问
"PETRE/TPAR/Presidio 呢"。改为对齐匿名轴的完整逐人电池。

## 臂与对比
- v6 臂 = `nec`(NEUTRALCLEAN=对应 `__v6min.json`,正典种子:MAD k8_s0 / CV k8_s0 / Enron k8_s1)。
- CONTRASTS:`nec-staab`(**P1 主对比**,对齐黑箱 ne−staab headline)、`nec-petre_k4`、`nec-tpar_t15`
  (Enron 另加 `nec-presidio`)、`nec-in`、`nec-cc`(后两者 = 次级/描述,补齐替代表格)。
- 单元集 = 复用 fc_v6min 包的单元(MAD 240u/16 簇、CV 106u/9 簇、Enron 112u/14 簇)——v6 判决波
  同尺,v6/staab/in/cc(部分)草稿缓存命中。

## 预注册判定(δ=.10 词典同 FC 标准)
- **替代主张 = 联合命题:v6 对每一个逐人臂都不 SIG-输**(SIG 赢或 TIE/UNDERPOWERED 皆可计入
  "不输",各对比按词典本名单独报)。联合方向使多比较对主张保守(比较越多越难过),不作校正;
  SIG 赢按对比单报,不合成"全赢"口号除非全部 SIG。
- 预期(登记):tpar(DP 降解)v6 SIG 赢;petre = 部分 no-op(52.6% 零改)≈ in 档 → nec-petre 可能
  TIE/UNDERPOWERED,如实报;staab 预期复现 ne−staab 方向(SIG 赢)。
- **kill**:v6 对任一逐人臂 SIG-输 → "效用格替代"该家降级,归因(卡内容 vs punt 通道,V2b 仪器)。
- 电池每波先行(pad/fmt/cut/self,含 in 臂无需 PROBE_ARM);fc_status 覆盖纪律;判官 sonnet 子代理
  波 ≤8,显式 model:sonnet。

## 登记代码改动
mad_fc_export.py / enron_fc_export.py 的 CARD 映射加 `petre_k4`/`tpar_t15`(Enron 加 `presidio`)
lambda(读各自 step2 文件既有键;CV 导出器已有全臂,零改动)。默认 CONTRASTS 不动;改后先以
原 env 复导一遍 fc_v6min 包核对 byte 一致再排新包。

## 费用闸
先 COST=1 报价;预算 ~$0.7(新起草 ≈ MAD petre/tpar/cc_s0 ×240 + Enron petre/tpar/presidio ×112),
超 2× 停下对齐。
**[R3′ 评审修正 2026-07-15,PROCEED-WITH-FIXES,跑前落实]**
- MAJOR-1:Enron fc_v6min 实为 240 单元/30 专家(112=池化臂去重算术,誤用于逐人臂)→ 新起草
  预算改为 MAD ×240×3 + Enron ×240×3 ≈ 1440 drafts ~$0.8-1.0;COST=1 实价为准。
- MAJOR-2(else 分支):**nec−staab 若非 SIG 赢(TIE/UNDERPOWERED)→ headline 降级为
  "v6 不输逐人去标识",不用比较级赢的措辞**;kill(SIG 输)条款不变。
- MINOR-3:MAD 必须显式 SEED=0(Enron 默认 s1 / CV 硬编码 s0 已匹配);byte 一致 = 真 diff
  (meta+config+抽批),不是目检统计。
- MINOR-9(顺手,不阻塞):enron 导出器 PERCLUSTER 无条件含 cc 的不一致留档,不改行为。
- CLEARED 备案:逐人臂三家 step2 键集与 staab 完全一致且按作者索引(无簇代表 bug);staab/in
  草稿跨种子缓存命中成立;CV 零改动零新起草;电池 PROBE_ARM 无需覆盖(含 in 臂)。

---

# R4 附录(任务 #128):gpt-5.4 第二攻击者复跑 v6 多种子 2AFC(2026-07-16 预注册)

**目的**:R1 的 v6 匿名判决全部出自 sonnet 单攻击者。复用 A2 模式(`gpt54_2afc_fixed_pkg/`),把
R1 的 **byte-identical** 2AFC 包交 gpt-5.4(用户 Codex 派发)重答,回答两问:
① v6 的匿名判决是否攻击者谱系无关;② **Enron (B)-泄漏 / CV 残余是 sonnet 波方差还是真残余**
(gpt-5.4 在完全相同的 pair 上独立复测 = 对 R1 残余的第二读数)。

## 包与来源(只打包 batch_*.json + meta.json + sys.txt,绝不带 sonnet 的 ans)
`gpt54_v6_pkg/`,builder = `scripts/build_gpt54_v6_pkg.py`(重批到 ~40 items/batch,prompt 逐字节不动):
- mad: s0=`results/mad/2afc_v6min`(正典)、s1/s2=`…_s1/_s2` — 768 pair/种子
- enron: s1=`results/enron/2afc_v6min`(正典)、s0/s2=`…_s0/_s2` — 696 pair/种子
- cv: s0=`results/se/2afc_v6min`(正典)、s1=`…_s1` — 462 pair/种子;**s2 无(卡集 R1 建卡门作废,
  README 如实说明,不是漏打包)**
共 5,316 pair,132 批(MAD 57/Enron 51/CV 24)。三通道全在包内:indiv(阳性对照)/ neutral(黑箱参照)/ conspf(评分时改名 v6)。
通道归属只在 meta.json(答案钥),worker 只见 pid+prompt。

## 预注册判定(scorer = `gpt54_v6_pkg/score_gpt54_v6.py`,逻辑对齐 cmd_multiseed_pool MSMODE=v6)
- **GATE(本次实测,不再引用旧包)**:gpt-5.4 池化 indiv 每家 CI-lo>0.5(硬门,不过则该家 v6 判决
  不可解读);强门 = acc ≥ sonnet 池化 indiv(.700/.662/.721 MAD/Enron/CV),只过硬门未过强门 →
  可解读但标注 weaker-attacker。
- **P1 绝对**:v6 池化 (B)(seed,card) + (C)(person) 双 bootstrap,δ=.10 认证 = 双 up95<.60 且双
  CI-lo≤.5(与 R1 同式)。sonnet 基线:MAD 认证 ✓ / Enron (B) LEAK [.506,.585] / CV 未认证 .568。
- **P2 相对(承重)**:配对差 ne−v6((seed,card) 簇配对 bootstrap)CI 含 0 = 不变差。sonnet:
  −.013/−.025/+.024 全含 0。
- **P3 归因读数**:按种子打印 v6 acc——Enron 若 gpt-5.4 也是 s2 波拉高 → 真残余(输入侧);若三
  种子皆 ∋.5 → 支持 sonnet 波方差解释。CV 同理看 s1。方向性描述,不做推断检验。
- **诚实条款**:gpt-5.4 若把 v6 打成双 pooler LEAK → 真匿名问题,按本名报,不以"攻击者更强"洗白;
  GATE FAIL → 该家不解读,如实报"本轮不加强主张"。
- 无 new-seeds-only 次级估计:R1 设它是因正典波 sonnet 先见;gpt-5.4 对全部波皆首见,无预选偏倚。

## 干跑/核对(交付前,$0)
① byte 校验:包内 prompt 集合 = 源包逐字节(按 (seed,pid) 全量 diff);meta 原样;计数 768/696/462×。
② 仪器校验:把 sonnet 的 ans 灌进包副本(scratch,不进交付物)跑 scorer → 必须复现 R1 数
(acc 精确一致,CI/up95 差 ≤ bootstrap 抖动)。
③ 交付 = 包 + README_FOR_CODEX.md + 用户可直接复制的 Codex 指令与评分命令。费用:本侧 $0,
gpt-5.4 侧用户 Codex。

**[R4 评审修正 2026-07-16,SHIP-WITH-FIXES,交付前落实]**(sonnet 对抗评审,评审员在 scratch 副本
实跑了 builder+scorer 并灌 sonnet ans 验证)
- MAJOR-1:scorer 配对差 diff() 误用全局池化均值 → 改为**等权簇均值**(每 (seed,card) 先算簇内
  ne−v6 再平均,与 v6_paired_diff.py 同式)。MAD 簇平衡碰巧一致掩盖了 bug;Enron/CV 修后精确复现
  R1 数(−.0253/+.0238),修前漂移(−.0273/+.0260,不翻含 0 判决但违反干跑复现门)。
- MAJOR-2:neutral 通道不打 CERTIFIED_ANON(cmd_multiseed_pool MSMODE=v6 的 REF_CHANS 语义,
  A1 拥有 ne 认证)→ scorer 打 REFERENCE-ONLY 标,README 加解读条(CV neutral LEAK = 已披露残余,
  非新发现)。
- MINOR:meta 缺失守卫;<2 簇时不崩;sys.txt 从正典种子取 + 断言各种子逐字节一致;pid 集合与 meta
  键集合相等断言;附录 "~133 批" 钉准为 132。
- CLEARED(评审员实证):8 源包 prompt/meta 逐字节打包一致;基线抽取数值精确(indiv .7005/.6624/
  .7208、v6 池化 .526/.546/.568、配对差全对);boot_by/certify 与 A2 模板逐字节同;包内无 ans 泄漏;
  cv/s2 确实不存在(与建卡门作废一致);README 全部数字与代码/JSON 对账通过。

---

# R11 附录(任务 #135):CV 多种子 FC —— nec−tpar 非劣冲刺(2026-07-16 预注册)

**动机**:论述改为"不损失效用前提下的最强匿名"后,效用格按单侧非劣核对(CI-lo>.40),16 格中 13 格
已认证;唯一挡路 = CV nec−tpar_t15(.495 [.386,.596],9 簇墙,差 .014)。补法 = R2 同款多种子扩簇:
新导 s1 波(nec 用 k8_s1 v6 卡,per-person 臂草稿缓存命中),与 fc_v6r3 的 s0 波池化,(seed,cluster)
9→18 簇,预期 CI 半宽 .106→~.075。

## 设计
- 新包 `results/se/fc_v6s1`:`CONTRASTS=nec-tpar_t15,nec-staab` `SEED=1`
  `NEUTRALCLEAN=cmd_shared_cards_cv__v6min.json`(k8_s1_G* 键已确认在)`PROBE_ARM=nec`。
  同 106 单元/63 专家;电池(self/pad/fmt/cut 载于 nec 草稿)每波先行,不过则波作废换判官重判。
- 判官 = sonnet 免费子代理(显式 model 参数),波 ≤8,单批串行写 ans 纪律,fc_status 覆盖校验。

## 预注册判定(δ=.10)
- **P1(主)**:池化 nec−tpar_t15 胜率,行 = s0(fc_v6r3)+s1(fc_v6s1) 全部 contrast 项,聚类单元 =
  (seed, unit_cluster),20000 次聚类 bootstrap。**非劣认证 iff 95% CI-lo > .40**(排除"tpar 好过 v6
  ≥δ")。旁报双侧词典判决(SIG/TIE/UNDERPOWERED)。
- **S1(次)**:池化 nec−staab 同机器。若 CI-lo > .5 → CV headline 可升比较级(标注"2 种子池化估计",
  取代 R3′ else 分支措辞);否则维持 R3′ 措辞不变。
- **稳健性/选择透明**:本扩充是在看过 s0 后决定的(与 R1 正典波先见同构)→ 旁报 s1-only 估计
  (9 簇,预期 underpowered,只作方向核对);tpar 的入选理由是功效不是方向(点值 .495≈null,选择偏
  倚极小),staab 次级按方向入选 → 池化估计带乐观选择偏倚,如实标注。分析计划在任何 s1 数据存在前
  冻结,不看中间数、不加对比、不动 δ、不建第三种子。
- **kill/else**:池化 CI-lo ≤ .40 → tpar 格按本名报"非劣未认证",CV 论述回退 staab 主对比;停。

## 登记代码改动
① `cv_fc_export.py`:`SEED = int(os.environ.get("SEED","0"))`(正典行为不变);② ne/cc/nec 卡存在性
断言条件化到**实际起草的臂**(CV concat 文件只有 k8_s0,本包不起草 ne/cc,原无条件断言会误炸);
③ 新分组清单 `cv_groups_k8_s1.json` 由导出冻结,并与 `results/se/2afc_v6min_s1/meta.json` 的
card→member 集合交叉核对(必须与 R1 的 s1 卡分组一致,否则 nec 草稿配错卡);④ 新池化评分器
`scripts/fc_multiseed_pool.py`(通用:多包+tag,聚类=(tag,unit_cluster),非劣+词典双判决;R2 复用)。
**Byte-repro 门**:改动后以 R3′ 原 env 复导 CV 包到 scratch BATCHDIR,与 results/se/fc_v6r3 逐字节
diff 一致(证明①②在 SEED=0 行为中立;绝不原地复导——导出器的 stale 机制会动真 ans)。

## 费用闸
COST=1 报价先行;预算 ≤$0.3(nec_s1 106 草稿 + 重采样;staab/tpar/电池全缓存或 $0),超 2× 停。
**[R11 评审修正 2026-07-16,SHIP-WITH-FIXES,起草前落实]**
- MAJOR-1(共同主判据):tpar 草稿跨波逐字节相同(缓存)⇒ 同一单元的 s0/s1 行共享 y 侧文本与题目
  难度,(seed,cluster) bootstrap 会把这份真相关当独立 → 低估 SE。修:**池化估计双聚类共同主判据**
  ——(seed,cluster) 与 **person(expert 跨波)** 两种聚类的 CI-lo 都 > .40 才认证非劣(与 2AFC 的
  B/C 双 pooler 同构)。
- MAJOR-2:两包 pid 命名空间冲突(C{ci}… 的 ci 是包内序号,s0 的 C0*=nec-staab,s1 的 C0*=
  nec-tpar)→ 池化工具**逐包解析、按 (tag,pid) 定位、只在解析后的 (unit,wave) 胜负值层合并**,
  绝不跨包合并裸 pid 字典。
- MAJOR-3:分组清单首跑即冻结 = 无对照的空检查 → 交叉核对改为**导出器内无条件硬断言**(SEED≠0 时
  必须存在 results/se/2afc_v6min_s{SEED}/meta.json,card→member 集合必须 ⊆ 导出器分组,任何不合即
  raise),先于清单写盘。
- MAJOR-4(幸存者偏差如实报):CV 只有 2/3 种子过建卡门(s2 以 7.4% 丢行被废,被废的恰是改动更重的
  卡集)→ 结果报告必须带一句:"池化估计以通过建卡门的 2/3 种子为条件;该选择对效用的方向未确证,
  但 plausibly 利好 nec"。
- MAJOR-5(次级降格):nec−staab 按方向入选 + 提功效复测 = 教科书式选择放大 → **S1 永远只作标注的
  敏感性分析,无论结果如何不升 CV headline**(撤销原"可升比较级"条款)。
- MINOR-6:条件化断言基于 contrasts 推导的臂集合(arms 变量在断言点尚未定义,勿引 NameError)。
- MINOR-7(时间线钉死):单侧非劣重构源自论述目标变更(用户指示,2026-07-16,"效用只需不损失"),
  先于逐格核对而非因某格失败而设;检验用 95% CI 下界,比标准 TOST(90% CI)更保守。

---

# R5 附录(任务 #129):qwen3.7-max 跨判官重判 v6 FC 包(2026-07-16 预注册)

**目的**:v6 时代全部效用判决(R3′ 替代表 14 格、ne−nec 不折损 TIE、R11 tpar 非劣)出自 sonnet
单判官。B2 模式(`mad_fc_judge_qwen.py`,BATCHDIR 通用,ans_qwen_*,非思考模式 t=0)只换判官重判,
回答"判决是否 sonnet 口味"。

## 范围(全 v6 FC 包)
fc_v6r3 ×3(MAD 86/Enron 95/CV 44 批)+ fc_v6min ×3(ne−nec 承重)+ fc_v6s1(R11 波)。

## 预注册判定
- 每包 qwen 自己的电池必须过(cv_fc_score ONLY=qwen 的 battery gate;不过 → 该包 qwen 无判决,
  只报"qwen 电池不过",不推翻 sonnet)。
- **复现判据(按对比逐格)**:方向 + 词典档位对照 sonnet 本名报。**kill = 任何 sonnet-SIG 格被
  qwen 反向 SIG** → 该格判"判官依赖",headline 降级;TIE→UNDERPOWERED / SIG→方向利 = 功效退化,
  如实报不算翻。R11 池化非劣在 qwen 下复算(fc_multiseed_pool 增 ONLY env,登记改动)。
- 本轮为稳健性 sidebar,不产生任何新主张、不改正典判决文件。
- 费用闸:逐包 COST 报价先行;任务预算 ~$1-2,**报价超 2×($4)即停,与用户对齐子集方案**
  (可选:只 fc_v6r3 headline 格 / BATCHES 子集低功效方向核对)。
**[R5 费用核准 2026-07-16]** COST 实报 $10.4(7 包合计;fc_v6r3 $7.2 + fc_v6min $1.9 + fc_v6s1 $1.4),
超任务标价 ~$1-2 → 按闸停下询问,用户核准**全量**。
**[R5 ops 改动登记 2026-07-16]** ① `fc_multiseed_pool.py` 增 ONLY env(单判官复算 R11 池化);
② `cv_fc_score.py` 输出名 ONLY 感知(`_fc_summary_qwen.json`,防覆盖正典 sonnet 汇总——B2 惯例);
③ R11 池化在 ONLY=qwen 下的电池门:load_pack 读的是正典(sonnet)summary 的 battery_pass,qwen 自身
电池由各包 `_fc_summary_qwen.json` 的 battery gate 另行把守,两者都过才引用 qwen 池化数。

---

# R2 附录(任务 #126):Enron ne−v6 多种子 FC —— 双侧 TIE 冲刺(2026-07-16 预注册)

**动机**:Enron ne−nec 正典 .460 [.388,.531],sMDE .102 差 .002 未达双侧 TIE(14 簇墙);单侧非劣
已成立(CI-hi .531 < .60)。目标 = 把"消毒不折损"的 Enron 格从"单侧非劣+方向反超"升到与 MAD/CV
同档的**双侧 TIE 认证**。扩簇 = R11 同款:新导 s0/s2 波,(seed,cluster) 14→42 簇。

## 设计
- 新包 `results/enron/fc_v6s0`(SEED=0)与 `fc_v6s2`(SEED=2):`CONTRASTS=ne-nec`
  `NEUTRALCLEAN=cmd_shared_cards__v6min.json` `PROBE_ARM=ne` `NEXPERT=30`(与正典 fc_v6min 全同)。
  **两臂均随种子变**(ne_s{seed} 黑箱卡 + nec_s{seed} v6 卡,k8_s0/s2 键已确认在)→ 每波 2×240=480
  新草稿,无缓存命中。电池每波先行,不过则该波作废换判官。判官 sonnet 子代理波 ≤8,fc_status 纪律。
- Enron 与 CV 不同:**3/3 种子全部通过建卡门**(s0 3.1%/s1 3.5%→修正为实测/s2 2.1% 丢行),
  无幸存者选择,如实注明。

## 预注册判定(δ=.10)
- **P1(主)**:池化 ne−nec over s1(fc_v6min,含既有 r1+r2 双判)+ s0 + s2,(B)=(seed,cluster)
  42 簇。**TIE 认证 iff (B) 双侧 equivalent && powered**(90% z-CI ⊂ [.40,.60] 且 sMDE<δ)——与全部
  既有正典 TIE(MAD .523/CV .519)同估计器族(pooling_cluster 聚类)。
- **S1(共同稳健)**:(C)=expert 跨波聚类(30 专家)须 ①无任一方向 SIG ②CI-lo > .40。(C) 若仅
  TIE-underpowered 不否决 P1(正典 TIE 均单 pooler;R2 的跨波复用刺激相关弱于 R11——两臂草稿逐波
  全新,共享仅题目难度,由 (C) 吸收),如实旁报。
- replicate 混合:s1 波 unit 值 = 双序 × (r1,r2) 均值(与 cv_fc_score agg 同式),新波 r1。
- **kill/else**:池化 (B) SIG 任一方向 → 按本名报(ne 赢 = 消毒折损为真,adverse 照报;nec 赢 =
  点反超升级),不授 TIE;仍 UNDERPOWERED → 报 sMDE 收口。**不建第 4 波**(3 种子已全)。选择透明:
  s1 波先见(canonical),s0/s2 全新;方向预期(登记):点值落 .46–.52,TIE 概率中高。

## 登记代码改动
① `enron_fc_export.py` 卡存在断言条件化到实际起草臂(concat 基础卡 k8 仅 s1,本包不起草 cc,
原无条件断言在 SEED=0/2 误炸——与 R11 CV 同雷);② SEED≠1 时 2AFC 审计轨迹交叉核对**硬断言**
(`results/enron/2afc_v6min_s{SEED}/meta.json` 的 card→member ⊆ 导出分组,先于清单写盘,R11
MAJOR-3 同款);③ byte-repro 门:改动后以正典 env(NEXPERT=30 CONTRASTS=ne-nec PROBE_ARM=ne
NEUTRALCLEAN=…,SEED 缺省)复导 fc_v6min 到 scratch BATCHDIR,与 results/enron/fc_v6min 逐字节
diff(绝不原地)。池化 = `fc_multiseed_pool.py` 原样(PACKS 三包,CONTRASTS=ne-nec)。

## 费用闸
COST=1 先行;预估 960 新草稿(draft_tok 700)≈ $0.5–1.0;超 2× 停,与用户对齐。
**[R2 评审修正 2026-07-16,SHIP-WITH-FIXES,导出前落实]**(评审员实跑了 COST 复现崩溃 + 模拟三种子分组)
- MAJOR-1:concat 断言未条件化实锤(SEED=0 实跑 AssertionError)→ 修为按实际起草臂;staab 断言同款
  加守卫(MINOR-6,当前不炸只因 step2 按作者键控,属侥幸)。
- MAJOR-2(登记措辞纠正):Enron 专家子样本 = 对 SEED 依赖的簇轮询 → 三波专家集不同
  (s0∩s1=26/30、s0∩s2=21、s1∩s2=21;并集 41 人、三波共同仅 19)——与 CV 的静态 63 人单元文件
  不同。(B) 不受影响(逐波簇全覆盖);**(C) 实为 ~41 人不均权聚类,原文"30 专家"作废**,按工具
  实际打印的 n_clusters 报;(C) 维持非阻塞。
- MAJOR-3:`fc_multiseed_pool` 的 certified_noninferior 是 R11 的单侧语义,**不得**用作 R2 的 P1;
  工具增 `certified_tie_B` 字段(pooled_B 双侧 verdict = TIE)并打印两行,R2 headline 只认 TIE 字段。
  (反例已登记:m=.45 CI [.41,.49] 会被 noninf 布尔误判"CERTIFIED",而正确判决是 DIFFERENT-Y。)
- MAJOR-4:登记 **s0+s2-only 次级估计**(28 簇,新数据独立答"TIE 是否只靠先见波"),PACKS 子集直跑。
- MINOR-5:成本预估修正:ne/nec 为 PERCLUSTER 去重起草(2 臂×14 簇×8 任务=224 稿/波,共 448),
  非 960;预算方向更宽。
- MINOR-7:`ONLY=r1` 复池化登记为必跑稳健行(replicate 不对称对称性检查;工具 glob 行为已核)。
- MINOR-8:SEED≠1 交叉核对置于清单冻结**之前**;评审员已预验 s0/s2 与 2afc_v6min meta 0 不合。
- CLEARED:双臂随种子变不引入偏倚(配对同波,nec 由同波 ne 机械导出,共同位移相消);两处代码改动
  在 SEED=1 行为中立(byte-repro 预期平凡通过);卡文件三种子齐全;pid 隔离与 unit 均值公式复核无误。

---

# R12 附录(任务 #136):tpar 重建方差 spot-check(2026-07-16 预注册)

**动机**:R11 认证了 CV nec−tpar_t15 非劣,靠的是 s0+s1 两个**分组种子**波的池化。但逐人方法没有
分组轴——它们的真随机轴是**重建方差**,而 tpar(DP-Prompt t=1.5 温度改写)是全部逐人建卡器里
随机性最强的:整条臂立在每张卡**唯一一次冻结抽样**上(chat 按 (model,messages,temperature) 缓存)。
问题:R11 的非劣认证是不是"恰好这一抽"的侥幸?

## 设计
- **重建**:`sample_one(tpar_msgs(card), deepseek-chat, s=2, temperature=1.5, max_tokens=1100)`——
  messages 与原建逐字节相同,仅缓存行 `_sample` 不同 → **分布不变的独立第二抽样**(不是换 prompt、
  不是换温度)。77 卡合并为 `tpar_t15_r2` 入 cv_cmd_step2.json(merge-append,不动其他臂)。空改写
  (T=1.5 拒答)按 build_tpar 同款检查,任何空卡 → 臂不完整,停。
- **新 FC 包** `results/se/fc_v6tr2`:`CONTRASTS=nec-tpar_t15_r2,nec-tpar_t15` `SEED=0`(正典分组,
  nec=k8_s0 v6 卡,其草稿全缓存)`NEUTRALCLEAN=cmd_shared_cards_cv__v6min.json` `PROBE_ARM=nec`。
  锚对比 nec-tpar_t15 同包 = 同波同判官,草稿全缓存 $0,用于把"重建效应"与"波效应"隔离。
- 判官 = sonnet 免费子代理(显式 model),波 ≤8,单批串行写 ans,fc_status+QUARANTINE 覆盖校验,
  电池先行(不过则波作废换判官重判)。

## 预注册判定(δ=.10)
- **P1(主,认证稳健性)**:三波池化 nec−tpar——行 = fc_v6r3(s0) + fc_v6s1(s1) 的 nec−tpar_t15
  + fc_v6tr2 的 nec−tpar_t15_r2(ALIAS 归一臂名),(B)(wave,cluster) 27 簇与 (C)expert 双聚类
  **CI-lo 都 > .40** → "R11 非劣认证在纳入独立重建波后维持"。端点字段 = certified_noninferior
  (这是非劣主张;R2 的 certified_tie_B 不适用)。旁报双侧词典判决。
- **否决条款**(防池化被旧波抬走):r2 波单独 95% CI-hi < .40(新波内 tpar_r2 显著超 nec 逾 δ)→
  无论池化结果如何**不得**声称稳健,按本名报"重建波单独显著劣于非劣线"。
- **S1(重建效应,纯描述)**:同波配对 win(nec−tpar_r2) − win(nec−tpar_r1) 逐单元差,cluster
  bootstrap CI(9 簇)。无论大小不升不降 headline——它测的是 tpar 重建方差的量级,本名入报告。
- **S2(选择透明)**:r2 波自身行(未被选择污染的新数据)旁报;s0 波先见、r1 臂先见照注。
- **稳健**:ONLY=r1 复池化(旧波有 qwen 复判行、新波只有 r1,replicate 不对称检查;默认 P1 与 R2
  同款 = 全 replicate 池化)。
- **kill/else**:P1 任一聚类 CI-lo ≤ .40 → 按本名报"R11 认证对 tpar 重建不稳健",CV tpar 格降级
  为"仅对 r1 建卡非劣";停。**范围钉死**:单次重建(s=2);分析计划在任何 r2 数据存在前冻结;
  不看中间数、不加建 r3、不改 δ、不换判官。

## 登记代码改动
① `enron_tpar.py`:抽纯函数 `tpar_msgs(card)`,`tpar_card` 改用之(行为不变);
② 新 `scripts/cv_tpar_rebuild.py`:**先只读直查 sqlite**(src.llm._key)验证原 tpar_t15 缓存行
  逐字节 == cv_cmd_step2.json 臂(CV 77/77;Enron step2_cards_full 116/116 同查,纯读 $0)——证明
  msgs 未漂移且原抽样仍在缓存;然后 sample_one(s=2) 建 r2 臂 merge-append;
③ `cv_fc_export.py`:CARD 增 tpar_t15_r2;臂存在性断言循环扩到 (_used ∩ step2 键) ∪ 原三臂;
④ `fc_multiseed_pool.py`:增 ALIAS env(行加载时臂重命名 "tpar_t15_r2=tpar_t15");未设置行为不变;
⑤ 新 `scripts/r12_anchor_diff.py`:S1 的同波配对差(meta+ans 直读,cluster_mean_ci)。

## Byte-repro 门(改动后、起草前)
- 门A = ②的缓存直查(77+116 全对上才继续;任何 miss/不等 = msg 漂移或缓存丢失,停下查因);
- 门B:R3′ 原 env 复导 CV 包到 scratch BATCHDIR,与 results/se/fc_v6r3 48 文件逐字节 diff 一致
  (绝不原地复导);
- 门C:R11 原 env + ONLY=r1 复算池化到 scratch OUTFILE,与 results/cv_fc_multiseed_pool_d10.json
  内容一致(d10.json 算于 R5 加 ans_qwen 之前,ONLY=r1 恰好复现当时的行集,兼验 ALIAS 改动中立)。

## 费用闸
COST=1 报价先行;预算 ≤$0.3(77 重建改写 ~$0.08 + 106 tpar_r2 草稿 ~$0.1;nec/tpar_r1/电池全缓存
命中,判官 $0),超 2× 停并对齐。

**[R12 评审修正 2026-07-16,sonnet 对抗评审,SHIP-WITH-FIXES,起草前落实]**
- MAJOR-1(新 x 侧相关):fc_v6tr2 复用 SEED=0 → 其 nec 草稿与 fc_v6r3 逐字节同、分区同,
  (B) 把 s0|Cj 与 r2|Cj 当独立(27 簇里 18 个不是独立证据),(C) 比共享资源(整簇一张卡)更细也
  吸不掉。修:`fc_multiseed_pool` 增 **TAGGROUP env → (D) 分区聚类**(同分区波共簇:s0,r2→p0|Cj,
  s1→p1|Cj,共 18 簇),认证改为 **(B)(C)(D) 三聚类 CI-lo 都 > .40**。
- MAJOR-2(ALIAS 混锚):包内既有真 nec-tpar_t15(锚)又有改名后的 r2 行,笼统改名会把两者按同键
  混平均 → 锚悄悄进 P1。修:**DROP + ALIAS 都按包定界**(`DROP="r2:nec-tpar_t15"` 先删锚行、
  `ALIAS="r2:nec-tpar_t15_r2=nec-tpar_t15"` 再改名),改名目标键已存在即 raise,全部动作响亮打印。
- MAJOR-3(否决太弱/强诊断被禁赛):CI-hi<.40 的否决只逮灾难级;温和变差(r2 点值 .40-.45)会被
  两个 r1 旧波投票盖过,而唯一有功效逮它的 S1(配对差,共享 nec 噪声相消)却被注册成纯描述。修:
  **S1 升级为门**——若 S1 95% CI 排除 0 于有害侧**且**点值 ≤ −δ/2(−.05),则即使 P1 过也不得
  声称稳健,按 CERT-WITH-FLAG 报("池化仍非劣但检出实质重建效应,稳健性主张不成立");原 CI-hi<.40
  否决保留为灾难级 tripwire。
- MAJOR-4(r2 可能悄悄等于 r1):门A 只验原行完好,不验新抽真的不同;缓存键 bug 可能让 r2 塌回 r1
  而无门可逮。修:重建脚本**断言 77/77 与 r1 逐字节不等**(t=1.5 下 exact match = 缓存 bug),
  并报长度比 + token-Jaccard 摘要。
- MAJOR-5(全 replicate vs ONLY=r1 不一致无规则):修:**认证要求两口径都过**(默认全 replicate
  与 ONLY=r1 任一不过 = 未认证,两口径都本名报;与 R2 的 P1 口径保持一致同时消除挑数自由度)。
- MINOR-1:导出器臂断言循环用 step2.get(m,{}),未建臂给友好信息不裸 KeyError;循环扩到
  ({staab,tpar_t15,petre_k4} ∪ (_used∩step2 键))。
- MINOR-2:重建脚本空卡 **assert 停**(不承袭原建卡器只 warn 的行为);加 r1-vs-r2 长度/vacuity
  census 逮"非空但拒答"。
- MINOR-3:门A 键重构 temperature 必须是 float 1.5 非字符串(json.dumps 序列化不同;错配 fail-safe
  为 77 个假 MISS,仍登记)。
- MINOR-4(共 x 同批盲评风险):兄弟对比(同 unit 同 order 的 nec-tpar_r1/nec-tpar_r2 项)按现散列
  ~4% 概率同批。修:批散列键 y 去 `_r\d+$` 后缀归一 + rebuild 对比偏移 1 槽 → 兄弟项保证不同批;
  正典包无 _r 臂,行为不变(门B 兼验)。
- MINOR-5:r12_anchor_diff 以 meta 的 (x,y) 识别对比,不按 CONTRASTS 位置。
- MINOR-6:S1 报告注明配对差还含跨判官噪声(散批规则下兄弟项多为不同判官),CI 已吸收,措辞不夸大。
- 门C 修订:d10.json 写于 R2 加 certified_tie_B 之前 → 门C 从逐字节 diff 改为**数值全等 + 新增字段
  白名单(certified_tie_B;TAGGROUP 未设时无 pooled_D)**。
- CLEARED:pid 命名空间(工具按 meta x/y 解析)、sample_one 缓存键不撞原行、电池纯本地变换 $0、
  胜率符号(两对比 nec 皆列首)、S1 配对良定义(同 106 单元)、SEED=0 无新分组漂移风险、费用核算、
  端点字段纪律(certified_noninferior 非 certified_tie_B)、无 r3 加建。

---

# R6 附录(任务 #130):消毒器模型互换 —— qwen 重建 v6(MAD k8_s0,2026-07-16 预注册)

**动机**:V6 的三轴性质(词面 0%、匿名不变差、效用 TIE)全部出自 deepseek 改写器。问题:这些性质
住在**机制与五闸**里(设计主张),还是住在 deepseek 的改写特异性里?换 qwen3.7-max(非思考,同温
t=0.3、同 prompt、同闸门、同阈值)重建 MAD k8_s0 的 16 张 v6 卡,三轴全套复测。这是加固实验:
**任何结果都不撤正典 deepseek 判决**。

## 设计
- **构建**:`v5_sanitize.py STAGE=build EDIT=min DATASET=mad K=8 SEED=0 SAN_GEN=openrouter/qwen/qwen3.7-max
  TAG=v6min_qwen` → `data/20mad/cmd_shared_cards_mad__v6min_qwen.json`(+audit/stats 同 TAG;正典
  __v6min* 绝不触碰)。五闸/冻结 prompt/阈值零改动(_config 的 prompt_sha1 必须与正典 stats 相同)。
- **建卡门**(继承 V6/R1):丢行率 >5% = 臂作废 kill;stats 的 tiers/changed_frac/words 对照 deepseek 报。
- **词面**:`elemk_v3_gates MODE=lex CARDS=cmd_shared_cards_mad__v6min_qwen.json`(仪器独立复核)。
- **匿名**:`cr_2afc_export DATASET=mad KCL=8 SEED=0 GROUP=random CHANS=<正典包同款> M_NNEG=2 M_RNEG=0
  CONSPFC=cmd_shared_cards_mad__v6min_qwen.json BATCHDIR=results/mad/2afc_v6qwen`(neutral/indiv 通道
  = 同波参照,与正典 2afc_v6min 同构);sonnet 免费判官(显式 model,≤8/波,单批串行,fc_status 纪律
  ——2AFC 包用 ans 覆盖直查);`r6_2afc_certify.py`(新)出判决。
- **效用**:`mad_fc_export SEED=0 CONTRASTS=ne-nec NEUTRALCLEAN=cmd_shared_cards_mad__v6min_qwen.json
  PROBE_ARM=ne BATCHDIR=results/mad/fc_v6qwen`(240 单元,ne 草稿缓存命中,nec=v6qwen 新起草);
  电池先行不过则波作废;`cv_fc_score.py` 出词典判决。

## 预注册判定(δ=.10)
- **P1(效用非劣)**:ne−v6qwen 胜率 95% CI-lo > .40(16 簇 cluster bootstrap)→ "qwen 消毒不损失
  效用";伴报双侧词典(正典 deepseek 参照 .523 TIE sMDE .090)。
- **P2(匿名认证)**:v6qwen(conspf 通道)nneg (B)=(card) 与 (C)=(member) 双聚类 **up95 < .60 且
  CI-lo ≤ .5** → δ=.10 认证(R1 MAD 同判据);**GATE**:同波 indiv 阳性对照 CI-lo > .5 必须过,
  不过则波作废换判官。
- **P3(不变差)**:同波配对差 neutral−conspf(逐簇等权,R4 模板)CI ∋ 0。
- **P4(词面)**:MODE=lex 0 行撞成员文本(>0 按本名报行数)。
- **kill/else**:丢行 >5% → "qwen 消毒器在 MAD 不能安全重建"本名报,停(不调阈不改 prompt 不换模型
  重试);P1/P2/P3/P4 任一失败 → "V6 该轴性质部分依赖改写器模型"按轴本名报;全过 → "V6 性质属于
  机制不属于模型"(消毒器无关)。范围钉死:MAD k8_s0 单家单种子 spot-check;不跑 CV/Enron;分析
  计划在任何 qwen 卡存在前冻结。
- 交叉参照如实注:2AFC 的 neutral/indiv 与正典包同 env 重建(判官波不同=波方差存在);比较一律
  同波内做,不跨波比点值(R1 教训)。

## 登记代码改动
① `v5_sanitize.py`:`GEN = os.environ.get("SAN_GEN", "deepseek-chat")`;两处 chat 调用加
  `extra=SAN_EXTRA`(SAN_GEN 含 "qwen" → {"reasoning":{"enabled":False}},否则 None——与
  mad_fc_judge_qwen/B2 对齐);`_TAG = os.environ.get("TAG", 默认不变)`;
② 新 `scripts/r6_2afc_certify.py`:通用单包 2AFC 认证器(B=(card)/C=(member) 双聚类 bootstrap、
  conspf 认证判据、neutral−conspf 逐簇等权配对差、indiv GATE)——A2/R4 模板落地为可复用脚本;
③ 其余仪器零改动(CONSPFC/NEUTRALCLEAN/CARDS/PROBE_ARM 均已 env 化)。

## Byte-repro 门(改动后、花钱前)
- 门A:`STAGE=build EDIT=min TAG=v6min_repro`(默认 SAN_GEN=deepseek)重建 s0 → 16 个 k8_s0_G* 键
  与正典 __v6min.json 逐字节相等(全缓存命中 $0;证明 SAN_GEN/TAG 插管中立 + 管线缓存确定性)。
- 门B:`cr_2afc_export` 以 CONSPFC=正典 v6min 复导 scratch,与 results/mad/2afc_v6min 的
  batch/meta 逐字节 diff(零代码改动,纯环境漂移保险,build_pairs 纯构造 $0)。
- 门C:`mad_fc_export` 以正典 fc_v6min env 复导 scratch + diff(同上,草稿全缓存 $0)。

## 费用闸
`STAGE=cost SAN_GEN=qwen` 报价先行。**任务牌价 ~$0.3;若报价 >$0.6(2×)必须停下与用户对齐后再花**
(qwen3.7-max 单价高于 deepseek,预估改写 ~900 调用可能到 $0.5-0.8);FC nec 起草 240 稿(deepseek)
~$0.2;判官全 $0。跑中实际花费超报价 2× 停。

**[R6 评审修正 2026-07-16,sonnet 对抗评审,SHIP-WITH-FIXES,起草前落实]**
- MAJOR-1(注册错了导出器):正典 results/mad/2afc_v6min 是 `neutral_2afc_export.py` 导的
  (samples_only.txt 佐证 + V6_REPORT 命令行 + pid 散批方案吻合),不是 cr_2afc_export(两者 swap
  散列方案都不同)。修:匿名建包与门B 都改用 **neutral_2afc_export.py**,正典 env = DATASET=mad
  KCL=8 SEED=0 CHANS=indiv,neutral,conspf NEUTRALC=cmd_shared_cards_mad__neutral_fixed.json
  CONSPFC=<v6 卡文件> NBATCH=32;评分器 = score_2afc_summary.py。
- MAJOR-2(CARDS 约定错):elemk_v3_gates 的 CARDS 是裸标签列表(自动拼 _CARDBASE__<lab>.json)。
  修:`CARDS=v6min_qwen`。
- MAJOR-3(TAG 静默无操作路径):忘设 TAG 时 stage_build 的续跑守卫(16 个 s0 键已在正典文件)会
  0 调用"跑完",下游全测正典 deepseek 卡还以为在测 qwen。修:build 内**断言 SAN_GEN≠deepseek ⟹
  TAG∉{v6min,v5san}**;_config 增 "GEN" 字段(卡片自证生成模型,续跑断言兼防混建)。
- MAJOR-4(单侧当双侧报,R2 MAJOR-3 同款):P1 headline 必须同时报单侧非劣与双侧词典;若非劣过而
  双侧非 TIE,措辞固定为"非劣(单侧);TIE 未达",不得写成与正典 .523 TIE 同档。
- MAJOR-5(报价器价格表硬编码 deepseek):stage_cost 是 0.28/1.10 常数且按 V5 全行改写口径高估行数、
  低估单价。修:按 SAN_GEN 查价表(qwen3.7-max ~1.25/5.00 $/M)+ EDIT=min 时用正典 v6min stats 的
  实测调用数(非 verbatim 行 + retried)估算;报价仍是估计,跑中超报价 2× 停。
- MAJOR-6(embed 无缓存,"全缓存 $0"前提破):cmd_consensus_pool.embed 每次实调 API;保真门 cos 贴
  阈值的行可能因 embedding 浮点噪声翻 tier → 门A 可能非字节复现且与插管无关。修:①给 embed 加
  sqlite 缓存(键=(model,text),行为中立,今后可复跑);②门A 判据改为**字节相等 OR(差异行 ≤2% 且
  逐行核对:复现版行文仍全过五闸、差异可归因保真阈值边界翻转——报差异行数)**;其余差异 = STOP。
  embeddings 单价可忽略($0.02/M)。
- MINOR-1(tier 漂移 tripwire):qwen 的 (fallback+relaxed) 行占比超 deepseek 同数 +15pp →
  CERT-WITH-FLAG("闸门更常兜底,机制主张带旗")。
- MINOR-2(范围句):互换只覆盖**改写层**;被消毒的黑箱卡与元素抽取仍是 deepseek 产物,结论措辞
  限定为"消毒/改写层模型无关"。
- MINOR-3(功效注):单种子 (B)=(card) 16 簇,同 δ=.10 词典但非 R1 三波 48 簇同等统计强度,如实注。
- MINOR-4(掉行不对称注):逐簇报 qwen−deepseek 丢行差,任一簇 |Δ|>1 行在报告中点名(掉行可能
  机械利匿名而害效用,防对消假全过)。
- CLEARED:extra 入缓存键安全、保真嵌入模型与 GEN 无关、build_pairs 三通道共享同 (member,stranger)
  对(P3 配对差良定义)、元素输入两臂逐字节同(stage_build 只读 ELEMS_P 缓存)、REQ 正则无模型
  依赖、_config 现有字段防 prompt/阈值漂移。

**[R6b addendum 2026-07-16(任务 #137,用户拍板"换个模型,试试 qwen3.7plus")]**
R6 预留分支落地:改写器 = `openrouter/qwen/qwen3.7-plus`(非思考 extra 同款),TAG=v6min_qwenplus,
其余与 R6 注册逐字相同(冻结 prompt/五闸/阈值、建卡门丢行 ≤5% kill、P1–P4 判定、MAJOR-1..6 与
MINOR-1..4 修正全部继承;门A/B/C 本会话已过且零代码改动,不重跑)。BATCHDIR:
results/mad/2afc_v6qwenplus、results/mad/fc_v6qwenplus;认证输出 results/mad/r6b_2afc_certify.json。
解释框架预注册:R6(max 版)已 kill,R6b 无论结果不改 R6 判决;R6b 过全部四轴 → "机制非模型"主张
以 2/3 改写器成立(deepseek+plus 过,max 不过,按本名列);R6b 也 kill → 主张收紧为"仅 deepseek
实证可安全重建,闸门层仍模型无关"。费用:STAGE=cost 报价先行(价表无 plus 条目时用保守默认
1.25/5.00 高估),预算 ≤$0.6,超 2× 停;FC nec 起草 ~$0.2(deepseek)。

**[R6c addendum 2026-07-16(任务 #138,用户拍板"试试 moonshotai/kimi-k2.6 以及 grok4.2,两个并行")]**
第三/四臂:`openrouter/moonshotai/kimi-k2.6`(TAG=v6min_kimi26)与 `openrouter/x-ai/grok-4.2`
(TAG=v6min_grok42),并行构建(不同 TAG 文件无碰撞;sqlite 缓存跨进程 timeout 容忍)。其余与 R6
注册逐字相同(判据/kill/P1-P4/修正继承)。登记小改:① SAN_EXTRA 非思考名单扩 kimi/grok(改写器
同条件 = 全部非思考,与 deepseek-chat 对齐);② stage_cost 价表补两条(注册时近似牌价,2× 停闸
兜底)。烟测先行(每模型 1 次微调用验 ID 可解析)。解释框架:任一臂过建卡门 → 下游三轴全套,
"机制非模型"以 2/N 成立;双 kill → "仅 deepseek 实证"限定进一步坐实,且失败模式解剖照报。
总预算 ≤$1.5(grok 单价高),报价先行,超 2× 停。
[R6c 更正:x-ai/grok-4.2 非法 ID;OpenRouter 实存 x-ai/grok-4.20(牌价 1.25/2.50),按用户'4.2'意图取 grok-4.20,TAG=v6min_grok420;kimi-k2.6 烟测通过。]

**[R6d addendum 2026-07-18(任务 #140,用户拍板"R6 最后再试两个模型:GLM5.1 以及 minimax2.7")]**
第五/六臂:`openrouter/z-ai/glm-5.1`(TAG=v6min_glm51)与 `openrouter/minimax/minimax-m2.7`
(TAG=v6min_minimax27),并行构建。其余与 R6 注册逐字相同(建卡门丢行≤5% kill、P1–P4 判定、
MAJOR-1..6 / MINOR-1..4 修正、解释框架全部继承;门A/B/C 零代码改动不重跑——本轮改动仅名单与价表,
与 R6c 同类)。烟测已过:两 ID 均可解析(glm-5.1 牌价 0.97/3.04、minimax-m2.7 0.25/1.00 $/M,
2026-07-18 取自 OpenRouter 清单)。登记小改:① SAN_EXTRA 非思考名单扩 "glm"(glm-5.1 非思考
调用烟测通过);② stage_cost 价表补两条。
**登记条件偏离(烟测发现,任何卡存在前钉死)**:minimax-m2.7 端点**强制思考、不可关**
(reasoning:{enabled:False} → 400 "Reasoning is mandatory")→ 该臂以 extra=None(供应商默认思考)
跑,是 R6 系列首个思考型改写器;与"全臂非思考"条件不同,该臂结论**必须带旗按本名报**
("思考型改写器"),不得与非思考臂并成一句无限定的"模型无关"主张;思考 token 计入 out 计费 →
报价旁注膨胀不确定性(牌价低于 deepseek,绝对额可忽略;探针 3 发内容均正常回,cap=400 未饿死)。
报价先行,总预算 ≤$1.5(两臂建卡 + FC 起草),超 2× 停。R6d 后 R6 系列收口:共 6 个替换改写器
(qwen-max / qwen-plus / kimi-k2.6 / grok-4.20 / glm-5.1 / minimax-m2.7)。

**[R6d 判决登记 2026-07-18]** 双臂皆死建卡门,kill 分支触发,不进下游:GLM-5.1 丢行 57/606 =
9.4%(fid 55 = qwen 家族同款保真死,最难簇同 G8/G13);MiniMax-m2.7 丢行 418/606 = **69.0%**
(empty 413 = **协议不相容**新失败类:强制思考 × 冻结 cap=400,探针证需 ~8k 思考 token 才出
内容,措辞钉死"该端点在冻结协议下不可用"而非"不会最小编辑")。R6c 机制结论只增强:6 替换
改写器无一全过三难,deepseek 唯一。报价教训按本名:minimax 实际 ~$1.2–1.4 vs 报价 $0.07
(思考 token 计费 × 重试烧满 cap),思考端点今后必须探针实测每调用 token 再报价。R6d 合计
~$1.6。详见 `V6R6_SANITIZER_SWAP_FINDINGS.md` R6d 节。

**[R6e 诊断探针预注册 2026-07-18(用户问:deepseek 为什么行?五轮限制是不是太严?)]**
- **问题**:fid 死法家族(qwen-max/plus、glm-5.1)的建卡门 kill,是"轮数预算太紧"还是"模型技能"。
- **$0 解剖(先行,已跑)**:glm 57 丢行中 42 = cov>0.90 全行改写路由线(路由输入侧确定,模型
  无关)= 数学上无最小编辑解、须整行换写+零 6-gram+保 cos 的最难任务;deepseek 在同 57 行上
  rewrite 22 / **rewrite_relaxed(打捞地板 cos≥.65)20** / 自己也丢 11(其全部 20 丢行的 55%
  落在此集)/ 其他 4,mean retries 3.5;deepseek 全量接受行 retry 分布 {0:129,1:129,2:50,3:25,
  4:17,5:119,6+:44} = 它同样磨梯子,靠"磨到底+打捞地板够得着"取胜,不是一发干净。
- **探针**:`scripts/r6e_retry_probe.py` — 对目标臂 audit 的 dropped 行原封调用
  `v5_sanitize._rewrite_min`(**唯一偏离 = monkeypatch MAXRETRY 4→11,即每段 12 轮**;用户判据:
  超 12 轮不测=模型问题;prompt/五闸/阈值/温度/SAN_EXTRA 零改动);chat() 缓存使前 5 轮反馈链
  逐字节重放续接原判($0),第 6 轮起才是新采样;子集调用合法性:逐行独立,消息只依赖
  (orig, runs, fail 链)。臂:glm-5.1 全部 57 行;qwen-max 最难三簇 G8/G13/G11 的 27 行(家族
  第二点)。**minimax 排除**:死于空回协议(empty→empty 反馈链逐字节同 → 加轮只会重放缓存空答,
  假测)。glm 的 1 条 empty 链同理测不了,按本名注。
- **记录**:每行 rescue 与否、tier、由 retries+tier 反推的过门轮次(打捞档记耗尽档);未获救行
  最后一发的 fid(audit 不存,此处补测)。哨兵:outright pass 轮次 ≤5 = 缓存重放假设破 → 硬失败。
- **判定(描述性,预注册措辞)**:重算 12 轮丢行率(glm 全测故精确);≤5% 门 → "轮数是限制因素
  之一"成立(**正典协议不改**:deepseek 5 轮已 3.3%,轮数加倍的成本/延迟按本名注);仍 >5% →
  "五轮不是瓶颈,是模型技能"。任何结果不撤 R6 系列 kill(kill = 冻结协议下的判决,本探针 = 机制
  归因)。qwen 子样本只报子样本获救率,不外推全臂。
- **附**:haiku 子代理(用户指定)对失败样本定性解剖(orig vs 最后一发:改超在哪、语义丢在哪)。
- **花费**:~$0.4(glm ~$0.2 + qwen ~$0.2),超 2× 停;$0 解剖与 haiku 免费。probe 不写 data/,
  输出 `results/r6e_retry_probe_*.json`(内部文件,含成员重叠原行,不随卡发布)。

**[R6e 评审修正 + 判决登记 2026-07-18]** sonnet 评审确认重放论证成立;1 MAJOR 采纳:探针必须断言
PROBE_TAG 的 stats._config.GEN == 探针模型(否则缓存全 miss → 12 轮全新钱且哨兵抓不到假重放);
4 MINOR(GROUP 强制、获救行诊断字段置空、tier 白名单断言、计数化简)全采纳。**判决:预注册
"模型技能"分支触发——glm 12 轮获救 12/57(打捞地板擦线 9 / 正经过闸 3),重算丢行率 7.4% 仍
> 5% 门;qwen-max 最难三簇 3/27 全靠地板擦线;未获救行 fid 挤在 .60–.65 地板下方 = 每发系统性
略微超改,加倍抽样不改单发分布**。deepseek 归因四证据(结构难度模型无关 / 编辑经济性 full-rewrite
档 .683 vs glm .776 配对 / haiku 定性抽象化漂移+重作文签名 / 自我熟悉度假说未直测)登记于
`V6R6_SANITIZER_SWAP_FINDINGS.md` R6e 节。实际花费 ~$0.5;正典协议 MAXRETRY=4 不动。

---

# R6f 预注册:指令诱导编辑经济性 —— 统一 GUIDE 补丁下的改写器可移植性(#142,2026-07-18)

**动机(用户拍板)**:R6e 证明 deepseek 的技能 = 编辑经济性;且 V6 prompt 本是与 deepseek 共同
演化的 → R6 系列测的是"裸插",对替换模型有公平性缺口。R6f 问:**同一段冻结指令能否把该技能
诱导出来**。用户核准的目标措辞(带前提):**"改写器可移植,但需指令诱导"——仅当 ≥2 个替换模型
用逐字节相同的 GUIDE 全四轴通过才可主张;qwen 单独过 = 1/2 初证,不得提前用此措辞**。后续模型
(用户逐个拍板)必须逐字节复用同一 GUIDE,不许逐模型改写(改一字 = 措辞作废)。

**冻结 GUIDE(在任何补丁臂调用前钉死,逐字节,一发定胜负,不许对着门迭代;失败照报,修订需
用户重新核准的新注册)**:
> Editing strategy, applied strictly: make the SMALLEST change that satisfies the request. Break a
> forbidden word sequence by changing roughly one word in every stretch of six — swap a single
> content word for an EQUALLY specific synonym, change word order, or shift a clause boundary — and
> keep the sentence skeleton, clause order, and every word you are not forced to change. Never
> replace a domain-specific term with a broader or vaguer one; never formalize or 'improve' the
> writing; never re-author the line from scratch. Preserve numbers, thresholds, conditions, roles,
> and the advice's direction exactly.

**机制**:`SAN_GUIDE` env 置位时把 GUIDE 逐字节追加到 MIN_SYS 与 RW_SYS 两处(同一文本,统一性
要求);未置位逐字节中立(正典安全);断言 SAN_GUIDE ⟹ TAG 非正典;_config 增 GUIDE 戳
(prompt_sha1/min_prompt_sha1 本身已因追加而变 = 续跑防混)。**已知张力按本名登记**:RW_USR 原文
"completely fresh wording" 与 GUIDE"保骨架"存在拉扯,不改 RW_USR(那是正典冻结件),由对照臂
实证判断补丁是否有害。闸门/阈值/仪器零改动(裁判不换)。

**顺序与判定**:
① **deepseek+GUIDE 中立性对照**(TAG=v6min_dsguided,只跑建卡门):丢行 ≤5% 过;**不过 = R6f
  整体中止**(补丁判有害,按本名报,不修订不重试)——护住 qwen 的一发。伴报 Δ vs 正典 3.3%。
② **qwen3.7-max+GUIDE**(TAG=v6min_qwenguided,非思考 extra 同 R6):建卡门 ≤5%,一发定胜负。
③ 过门 → 下游三轴全套(R6 MAJOR-1 修正后的导出器:neutral_2afc_export.py 同波三通道 + sonnet
  判官显式 model ≤8/波;mad_fc_export NEUTRALCLEAN/PROBE_ARM=ne 240 单元 + 电池;P1-P4 判据与
  R6 注册逐字同)。任一轴败 → 按轴本名报"指令诱导救回建卡门但救不回该轴"。
**预埋观察点(过门后必测必报,不预设方向)**:kimi 前车 = 风格层匿名(效用 TIE 却 2AFC LEAK);
"保骨架更多"理论上可能抬 R7 模糊词面(τ=.90)残余 —— 有仪器,照测。
**措辞影响预登记**:若成立(≥2 模型),R6c/R6d/R6e 的"deepseek 唯一/技能不可移植"统一软化为
"裸插不可移植,技能可由统一指令诱导";若失败,"唯一"结论升级为因果证据。**R6 系列 kill 一个
不撤**(它们回答冻结协议问题)。
**花费**:ds 对照 ~$0.1 + qwen 建卡 ~$0.35(报价先行)+ 过门后下游 ~$0.5;超 2× 停。
**落地校验(跑前已过)**:SAN_GUIDE 未置位 prompt sha1 逐字节等于正典戳(rw 48c4f3471f94 /
min 121c6e809a98);置位两段均尾接 GUIDE、sha1 变(273c68f006bd / b5b9a484d65b = 续跑防混)。
**GUIDE sha1 = c0e58fdc7cd3(后续模型逐字节复用的锚,不一致 = 措辞作废)。**

**[R6f 中期登记 + 波2 增补 2026-07-18(波2 放出前钉死)]**
① ds+GUIDE 对照过门:4.79% ≤5%(正典 3.3%,+1.5pp 带旗;死法翻转 fid→lex 24/29 = "多保留"
的代价,补丁对 deepseek 非免费,正典不采用)。② qwen+GUIDE 建卡门 **0.99%**(裸插 11.2%,
全系列最低;fid 死 61→1);P4 词面 0.0%×2 ✓;P1 效用 ne−nec .512 [.462,.560] sMDE .070
**双侧 TIE 认证 ✓** 电池 PASS;P3 配对 neutral−conspf +.004 [−.039,+.051] ∋0 ✓;P2 绝对格
conspf .539 [.461,.606] up95 差 .006 未达,**同波 neutral 参照 .609 同超线** + 两判官自述跨题
侧信道 = 波偏强(R6c grok 同款形状,非 kimi 式臂内真漏)。
**波2 增补(判决前预承诺)**:同包 32 批复制至 `2afc_v6qwenguided_w2`,第二独立 sonnet 判官波;
**池化判定:无论波2 结果如何一律并入,双聚类((B)=card/(C)=member)bootstrap δ=.10,单波与
池化并报,不选波**(R2 破簇墙先例;判官免费 $0)。若池化仍未认证 → P2 按本名"未认证"定档,
匿名结论以 P3 为主、P2 如实列(R6c grok 公约),且用户措辞门("全四轴")按未达处理。

**[R6f qwen 臂判决登记 2026-07-18]** **全四轴通过(1/2 初证)**:建卡 0.99% / lex 0.0%×2 /
效用 .512 [.462,.560] TIE 认证(电池 PASS)/ **匿名池化 .521 (B)[.445,.592] (C)[.463,.580]
δ=.10 双聚类 CERTIFIED**(w1 .539 up95 .606 差 .006 未达 = 波强度;w2 solo .504 自行认证;
配对 w1 +.004 / w2 +.023 / 池化 +.014 全 ∋0;indiv 门 .703/.699/.701)。完整性:两波覆盖
768+768 脚本校验;防篡改扫描 1 批 conspf 5/5(32 批随机期望 ~1.2,与噪声一致),剔除敏感性
.517 [.441,.588] 认证不变;w2 判官提示"逐题独立"较 w1 更明确(w1 两判官自述跨题侧信道后收紧,
如实注);一 w2 判官发现包内 meta.json 含真值自述未用(历史协议如此,已立防漏后台任务,防篡改
扫描即为此而加)。**R6f-2 增补:第二模型 = glm-5.1(用户"没问题了再加模型";fid 家族另一员、
独立厂商;minimax 协议排除),GUIDE 逐字节同锚,TAG=v6min_glmguided,判据/流程与 qwen 臂逐字
同(建卡门 → 三轴 → 需要时波2 池化同预承诺)。

**[R6f 终局判决登记 2026-07-18]** glm+GUIDE:建卡 3.80% ✓ / lex 0.0%×2 ✓ / 效用 .508
[.469,.554] sMDE .061 TIE 认证 ✓ / P3 配对 −.004/+.039/池化 +.018 全 ∋0 ✓ / **P2 绝对格未认证**
(池化 .564 (B)[.479,.646];同波 neutral 参照自身 .582 [.516,.645] 读 LEAK = 波对偏热,四波
neutral 读数 .527–.602 波方差实证;grok 公约 P3 为主)。按预承诺不加第三波(wave-shopping 禁)。
**措辞判定(用户门,不放水):严格版"可移植但需统一指令诱导"(≥2 全四轴)未达——qwen 全四轴 ✓、
glm 差 P2 一格;论文措辞定为"改写器技能可由统一指令诱导:2/2 建卡+词面+效用+配对不变差,匿名
绝对认证 1/2(波依赖如实注)"**。防篡改扫描/覆盖/演化注见 V6R6_SANITIZER_SWAP_FINDINGS.md R6f
节。花费 ~$1.4;正典不采用 GUIDE;R6 kill 不撤。

**[R6f-3 增补 2026-07-18(用户拍板"试试kimi")]** 第三模型 = `openrouter/moonshotai/kimi-k2.6`
(TAG=v6min_kimiguided,非思考 extra 名单已含 kimi),GUIDE 逐字节同锚 c0e58fdc7cd3,判据/流程
与 qwen/glm 臂逐字同(建卡门 ≤5% → lex → 2AFC 严格版判官提示 + FC → 需要时波2 池化同预承诺,
禁 wave-shopping)。**本臂的机制赌注与前两臂不同**:kimi 裸插(R6c)过建卡门 4.6%、效用 .544
TIE(单侧差 .007 注),死于**臂内真实**风格层泄漏(conspf .590 [.535,.641] LEAK 而同波 neutral
.555 ∋.5 不漏)——若 GUIDE 的"保骨架最小扰动"洗掉 kimi 措辞指纹使匿名过,即"指令管住风格层"
的直接证据;若仍 LEAK(同波参照不漏),则"风格指纹不受编辑经济性指令控制"按本名报,措辞维持
1/2。**若全四轴过 → 严格版措辞凑齐 2/2(qwen+kimi),按用户预注册门生效**。解释框架预登记:
kimi 裸插与 +GUIDE 的效用/匿名比较一律同波内或按 R6c/R6f 各自波如实并排,不跨波比点值。
花费:建卡报价先行(~$0.3-0.4)+ FC 起草 ~$0.2,预算 ≤$1,超 2× 停。

**[R6f-3 判决登记 2026-07-18]** kill:丢行 89/606 = **14.69%**(裸插 4.6% 的 3 倍),lex 76/89
——kimi 被 GUIDE 推入"改不够"区,打不断禁串。风格指纹问题未获测量,不进下游。**机制结论:
GUIDE = 沿编辑经济性轴的恒定平移非回中信号(qwen/glm 过改→修复;ds 近优→略过头;kimi 靠大改
过活→毒药);deepseek 技能本质 = 校准**。kimi 两头堵死(裸插匿名死 / +GUIDE 建卡死)补完 R6c
案例。严格版措辞维持未达(qwen 唯一全四轴)。花费 ~$0.25;R6c kimi 裸插判决不撤。

---

# R13 预注册:全梯度 k 消融重扫(导师要求),MAD 双轴,k=2/4/6/8/10/12 on 正典 v6(#139,2026-07-17)

**问题**:正典 CMD(pool + V6 消毒)的匿名/效用两轴对 k 的剂量反应,全梯度实测(不再靠
ne 旧梯度 + 转移论证)。范围 = 用户拍板方案 A:仅 MAD,双轴。#132 R8(k4 spot)被本实验吸收。

## 0. 库存与缺口(盘点已做,$0)

| 层 | k2 | k4 | k6 | k8 | k10 | k12 |
|---|---|---|---|---|---|---|
| base shared | s0:64 | s0/1/2:32 | s0:21 | s0/1/2:16 | **s0:12** | **s0:10** |
| neutral(+fixed) | s0 | s0/1/2 | s0 | s0/1/2 | ✗ | ✗ |
| v6min | ✗ | ✗ | ✗ | s0/1/2 | ✗ | ✗ |

分组即 `cmd_gate.make_groups`(GROUP=random,余数组 <k 丢弃:k6 覆盖 126 人、k10/k12 覆盖 120 人
——与既有 k6 先例同法,预注册如实报覆盖数)。k16 base 在但超出用户网格,出界不做。

## 1. 建卡阶段(deepseek 正典,全冻结 prompt)

- **B1 neutral 合成 k10_s0+k12_s0(22 簇)**:新脚本 `r13_neutral_build.py` = a3_qwen_neutral_build
  的 deepseek 版(`mad_synth_utility.synth_neutral` 正典 prompt 逐字复用 + `cmd_fix_degenerate`
  退化检测/anti-copy 修复,MAX_TRIES 同参),差异仅:模型=deepseek-chat(不换)、K/SEED env、
  写入 = **追加进正典多 k 文件** `cmd_shared_cards_mad__neutral{,_fixed}.json`(key 含 k 无碰撞;
  写前备份 `.r13bak`;追加后断言:旧 key 集合逐字节不变)。
- **B2 v6 消毒 k∈{2,4,6,10,12} s0**:`v5_sanitize.py` EDIT=min,放宽 K 断言(R1 放宽 SEED 同款:
  `K in (2,4,6,8,10,12) and SEED in (0,1,2)`,非正典 (K,SEED) 打印显式横幅;key 含 k 无碰撞;
  prompt_sha1/五闸/阈值/重试梯全部不动),TAG 仍 v6min(GEN=deepseek 正典,resume 防呆断言不触发),
  输出追加进 `cmd_shared_cards_mad__v6min{,_audit,_stats}.json`(同款备份+旧 key 不变断言)。
- **建卡门(逐 k 档判)**:每 k 档丢行率 ≤5%(正典 k8_s0 = 3.3%)。某档超门 → 该档按本名 kill
  (不进该档下游),其余档照走;全档报丢行/词量比/tier 分布。
- 词面证书:每档消毒后跑独立 lex 仪器(与 R6c 同款,6-shingle vs 成员文本),要求 0%。

## 2. 匿名轴(5 波 2AFC,免费 sonnet 判官,$0)

每 k∈{2,4,6,10,12} 一波:`neutral_2afc_export.py` DATASET=mad KCL=k SEED=0
CHANS=indiv,neutral,conspf,NEUTRALC=neutral_fixed,CONSPFC=v6min(conspf 通道读 v6 卡,key 方案
同构,R6c 已验证此插法);M_NNEG=2 → 每通道 ~240-256 对,每波 ~720-768 对 12 批。
k8 点不重跑:直接引用 R1 的 s0 波(同导出器同通道结构)。判官纪律:批内亲判不委托、逐批串行、
波≤8 并发、fc 覆盖直查、全批重排队规则照旧。

评分:每波 `r6_2afc_certify.py`(GATECH=indiv 阳性对照须过,TEST=conspf 双聚类,REF=neutral
仅同波参照)。**预注册读法**:
- 主对象 = **同波配对 v6(k)−ne(k)**(波内相消,免疫跨波判官漂移;预期全部 CI∋0 = 消毒不改变剂量)
  + **v6(k) 绝对格剂量曲线**(预期低 k LEAK 单调向 k8 残余衰减——这是测量不是 kill,任何形状照报)。
- 跨波点值比较禁止下认证结论(R1 教训);曲线呈现时 ne(k) 同波参照曲线并排画,漂移可见。
- δ=.10 绝对认证仅在 k8(已有)与如果某低 k 意外达标时按本名报;低 k 预期不认证。

## 3. 效用轴(5 个 FC 包,起草 deepseek、判官免费 sonnet)

`mad_fc_export.py` 改动(预注册):K 由硬编码 8 → env(默认 8 不变);新增比较臂 `ne8` = 该 dev
所在 **k8_s0 分组**的 v6 卡(逐 dev 查表,与包分区无关,bootstrap 按 k8 簇);probe 臂 `nek` =
v6(k) 卡(NEUTRALCLEAN 式指到 v6min 文件、前缀 k{k}_s0)。每包 SEED=0(与 v6 卡对齐,R6c 先例)。
- **主对照**:nek−ne8 逐 unit 配对 FC(同 dev 同 bug 两张卡直接对决)。判定:词典本名
  (SIG/TIE/UNDERPOWERED);TIE 认证口径 = 双侧 90% CI ⊂[.40,.60] + sMDE<δ,δ=.10;双聚类
  (B)=k 分区簇 /(C)=k8 分区簇,同 R11 纪律两口径并报。
- **副对照(近零成本)**:nek−in(in 臂起草跨包同 (dev,bug) 全缓存 $0)→ 共同参照剂量曲线。
- 电池:pad/fmt/cut/self 照常(NPLA/NCUT 默认),电池不过 = 该包作废重看,不改判据。
- **功效诚实申报**:k10/k12 包 (B) 口径仅 12/10 簇,sMDE 预期 ≥.10 → 这两点按预期 UNDERPOWERED
  呈现(报点估+CI+sMDE),曲线主张靠 k2/k4/k6(64/32/21 簇)+ k8 锚;不为凑认证加判官波。
- in/staab 臂起草全缓存;新起草 = nek×5 档 + ne8×1 次(v6 k8_s0 若 R5/R124 已缓存则 $0,
  STAGE=cost 实测为准)。

## 4. 判决与解释框架(预注册,防事后挑数)

- 匿名:配对 v6−ne 全档 ∋0 → "消毒不改变 k 剂量,匿名来自聚合"实测钉死(R8 转移论证升级为直测)。
  任何档配对 CI 排除 0 → 按本名报"消毒×k 交互",方向照报,不撤 k8 正典。
- 效用:nek−ne8 在 k4/k6 预期 TIE;k2 允许 SIG 输(旧仪器时代 k2 显著变差,若复现 = 新旧仪器一致的
  真剂量效应,按本名报);k10/k12 UNDERPOWERED 口径。**旧 #33"k 近自由旋钮"结论由本实验取代**
  (那是死仪器时代的数字,R13 落地后文档里不再引用)。
- 不对称规则照旧:任何 kill/异常不撤 k8 正典判决。
- 曲线拟合仅描述性(报每档点估+CI;不做跨波推断检验;Δh∝k^−α 引用旧梯度时标"ne 旧仪器"来源)。

## 5. 成本与门(报价先行)

粗估:B1 合成 ~$0.1 + 退化修复 ~$0.05 + B2 消毒 ~139 卡/~5300 行 ~$0.9 + FC nek 起草 5×~$0.2
≈ **$2.0-2.5 总**(判官全免费;嵌入闸走 sqlite 缓存)。精确报价:B2 用 stage_cost(逐档)、FC 用
COST=1(逐包)跑完后汇总;超粗估 2×($5)停下对齐。三门照旧:门A 缓存重放(改 K 断言后正典 k8_s0
必须逐字节复现,证插管中立)、门B 导出 byte-repro(改 mad_fc_export 后正典包干跑重导逐字节对比)、
干跑先行。判官波错峰:2AFC 5 波先行(建卡完即可),FC 包随后,总墙钟 ~2-3 天。

## R13 评审修正(sonnet 对抗评审 PASS-WITH-FIXES,6 MAJOR + 4 MINOR,2026-07-17)

- **M1(隐藏前置,必崩)**:v5_sanitize 依赖 `elemk_elements_k{K}_s{SEED}.json`(词面闸 src_sh 本体),
  现仅有 k8。B2 前每档先跑 `elemk_build.py STAGE=extract DATASET=mad K=<k> SEED=0`(extract 逐人、
  chat 缓存键不含 K/SEED → 预期 $0 全缓存命中;干跑验证)。
- **M2(必崩)**:mad_fc_export 的 neutral/concat/staab 存在性断言无条件触发(concat 无 k2/k6/k10/k12,
  neutral 暂无 k10/k12)→ 照 `_use_nec` 写法按 CONTRASTS 实际用臂条件化。
- **M3(主对照重定,设计级)**:撤销"包内 ne8 跨分区臂"(cv_fc_score 单一 UCLU 聚类对跨分区臂统计无效,
  两随机分区的 join 退化)。改为:①主对照 = **每包 nek−in**(与正典 ne−staab 同构:池化臂 vs 逐人臂,
  单分区聚类完全成立,scorer 零改动);②补一个 **k8 锚点包**(K=8 SEED=0 的 nek−in,即 v6 k8 vs in,
  起草预期大量缓存命中);③跨 k 主张走**同 unit 配对 Δ(k)=d_unit(k)−d_unit(8)**(units 与分区无关、
  跨包逐字相同),推断用新独立脚本 `r13_fc_curve.py`:两因子聚类 bootstrap(k 分区与 k8 分区独立重采,
  CGM 式)为主,两个单因子口径作敏感性并报——**不动 cv_fc_score.py**。
- **M4(报价门失效)**:stage_cost 对非 k8 档因精确前缀匹配空集而虚报 $0 → 改为无精确样本时按已有
  EDIT=min 记录的每行调用速率外推,并打印显式"外推"横幅;静默 $0 视为 bug。
- **M5(建卡门口径)**:脚本自带汇总是跨档累积口径 → 每档判门用外部按 `k{K}_s0_` 前缀过滤的独立统计,
  不信控制台累积打印。
- **M6(静默降采样)**:FC 每包显式 `NEXPERT=0`(默认 30 会在 k2/k4 崩、k6-k12 静默抽 30 人)。
- MINOR:①放宽横幅措辞同时报 K/SEED;②`r13_neutral_build.py` 直接 import 函数不走 CLI main
  (绕过 `ck not in src` 静默门),且同时维护 neutral 与 neutral_fixed 两个正典文件(各自备份+旧 key
  逐字节不变断言);③r6_2afc_certify 的 paired 输出符号 = neutral−v6(正 = v6 更不可识别),报告
  显式标注;④CONSPFC 必须传全名 `cmd_shared_cards_mad__v6min.json`(否则静默回退默认 conspf 文件
  测错卡),运行配方写全名 + 导出后 config 回显核对。

修正后成本更新:FC 变 6 包(+k8 锚点,起草大头预期缓存命中),总粗估 **$2.2-2.7**,停闸线不变($5)。

### R13 库存勘误 + 报价(2026-07-17,$0 阶段完成)
勘误:§0"余数组丢弃"错——`make_groups` 把余数**摊进各组**(k10=12 组 sizes 10-11、k12=10 组
sizes 12-13、k6=21 组 6-7),三档皆 128/128 全覆盖,无人失覆盖(比注册假设更好,如实更正)。
报价实测($0 COST 模式):B1 合成 k10+k12 ≈ $0.10;B2 消毒 k2 $0.27 / k4 $0.15 / k6 $0.10
(M4 外推横幅正常),k10/k12 待 B1 后精报(按行数外推 ~$0.13);FC nec 臂起草待 COST=1 逐包精报
(粗估 ~$1.2-1.5)。合计走向 ~$2.0-2.3,在注册包络内,停闸线 $5 不变。

### R13 门检结果(2026-07-17)
门A PASS:改 K 断言后缓存重放重建 MAD k8_s0,16/16 卡逐字节复现正典 v6min(插管中立)。
M1 前置:elemk 元素抽取 k2/4/6/10/12 全部落盘,逐档统计与 k 无关且完全一致(mean 22.5/member)
= 底层 LLM 缓存命中,$0。
门B:直接对比正典包 fc(2026-07-13)发现 342/353 —— 解剖 = 10/10448 条 draft 的缓存内容漂移
(同 pid 同批次同槽位,文本为另一采样;batching/meta/sys 全一致)。隔离实验:反做 R13 三处编辑的
PRE13 版重导 → **与 R13 版 353/353 逐字节一致 = R13 编辑导出中立(门B 按意图 PASS)**。
正典包漂移为先存缓存血统问题(B1 日与今日之间 ~10 行缓存被重建),按本名登记:不影响已判的
正典 B1 判决(判官判的是当日包),影响的是"今日重导可复现性";R13 全部 FC 包为全新导出,与
今日缓存自洽。后续任何 byte-repro 门若再撞此 10 条,以本条为准,勿重复解剖。

### R13 预算改注(2026-07-17,用户拍板)
FC 报价实测超粗估(2048 稿/包,评审后设计的同 unit 配对 Δ(k) 要求全 1024 单元,而 v6 时代旧包
均为 240 单元省钱口径)→ 按停闸规则对齐,用户选 A:全 1024 单元 × 6 包(含 k8 锚点包升口径),
新预算 ~$5.5-6.5(帽 ~$8)。R3′ 的 nec−in(240 口径)保留作 k8 锚点 sanity 参照。词面证书五档
全 0.0%(黑箱参照 36-44%/73-85%)已实测;2AFC 五包 768 对已导出待判。

### R13 判决登记(2026-07-17,全文见 results/V6R13_K_GRADIENT_FINDINGS.md)
匿名:v6 全 6 档(含 k2)逐点 δ=.10 双聚类认证,配对 v6−ne 全档 ∋0,indiv 门全过 →
"低 k 必漏"预期(源自 ne 卡)被推翻;消毒杀风格通道 + 池化杀内容身份,叠加后全梯度到 chance。
效用:5/6 档 TIE 认证;k10 逐点 SIG(.450)但配对 Δ(k10) ∋0 且 k12 恢复 TIE → 无衰减趋势,
效用对 k 平坦(取代 #33 死仪器结论)。建卡门五档 1.9–3.3% 全过、词面证书全 0%。
覆盖:516 判官批 100%;花费 ~$3.3(核准 $5.5-6.5 内)。#132 R8 被吸收。

# R7 附录(任务 #131):模糊词面攻击者 + C1 跨发布链接 on v6(2026-07-17 预注册,评审前冻结稿)

## 动机(两个都是堵审稿问)
1. **"0% 精确 6-gram 证书可被同义改写绕过"**:词面证书只管逐字连串;R6c-kimi 实证了"词面 0% 但风格层 LEAK .590"的组合真实存在 → 超词面攻击者不是假想敌。R7 测**正典 deepseek v6**(其 δ=.10 绝对认证已过 ×2 攻击者)在"精确词面以上、LLM 风格以下"的**模糊词面通道**(嵌入近邻)的残余。
2. **"池化卡多发布可链接"**:正典 C1 只测了 ne(.914)与 concat(.974);v6 砍掉 ne 的逐字通道(rare6 基线 ne 波 .826)后链接应下降,决策架构通道保留 → 预期读法"更不可链接、非免疫"。

## E1 模糊词面攻击者(机械/确定性,无判官,~$0.05 嵌入)

**E1a 阈值扫描 census(描述性,证书的模糊化延伸)**
- 对象:正典 k8 卡 ne 与 v6,三家,(K,SEED) = elemk census 的 _CANON(MAD k8_s0 / CV k8_s0 / Enron k8_s1)。
- 每卡行(_card_lines 同源切分)算 max-cos(text-embedding-3-small,EB.embed sqlite 缓存)对两个参照:
  (i) **成员参照** = 本簇成员 elements ∪ aggro 行(census 同源);
  (ii) **陌生 null** = 等规模非成员参照(每簇 seeded 抽同数量陌生人,同构造)。
- τ ∈ {.70,.75,.80,.85,.90,.95} 六档曲线:行占比(max-cos≥τ);**主读数 = member−stranger 差**(纯"到处都像"的领域通语在两参照上对称,差才是泄漏信号)。
- 无判定词典(描述性);预期 v6 差→0、ne 差大;不预期不惩罚。

**E1b 机械模糊 2AFC(推断性,骑正典 G3 波配对集)**
- 配对集 = `results/{mad,se,enron}/2afc_v6min` 的 meta(chan ∈ indiv/neutral/conspf,conspf=v6;MAD 256×3 / CV 154×3 / Enron 232×3 对)。
- **门A(重建)**:A.build_pairs() 以各波原 env(DATASET/KCL/SEED/CHANS/NEUTRALC=neutral_fixed/CONSPFC=v6min/M_NNEG=2/M_RNEG=0)重建 + neutral_2afc_export._swap 同则,逐 pid 与包内 meta.json 六字段(chan/neg/card_id/member/stranger/member_slot)全等;不全等即停,不硬凑。
- 攻击者:对候选 cand ∈ {member, stranger},S(cand) = 卡行 max-cos 到 cand 的 ref 文本行(与 LLM 攻击者看到的同一份 _mref/_sref)后聚合;**聚合档:主 top5mean,备 max、mean。档位唯一由 indiv 阳性对照决定**(顺序 top5mean→max→mean,取第一个 indiv SIG>.5 的档为正式档;全败 → E1b 判 instrument-limited,只报 E1a),**不看 ne/v6 数字选档**。选 member 当 S(member)>S(stranger),平手记 .5。
- 读数:每通道 acc + card_id 聚类 bootstrap CI(score_2afc_summary 同款统计);配对差 ne−v6(同 (card_id,member,stranger) 单元,簇聚类)。
- 预期分支(全部登记,按本名报):
  (i) indiv SIG + ne SIG 漏 + v6 ∋.5 → "模糊词面通道存在于 ne、被消毒关闭";
  (ii) indiv SIG + ne ∋.5 → "机械模糊攻击者破不了任何池化卡 = 通道对池化本来就弱;证书辩护退回 E1a";
  (iii) v6 SIG 漏 → 如实报,与 LLM 攻击者认证并列为通道特异残余,不撤正典认证。

## E2 C1 跨发布链接 on v6(MAD k8 ×3 种子,判官免费)
- v6 卡 = R1 三种子(MAD s0/s1/s2 全过建卡门,48 键;CV 的 2/3 幸存者注**不适用**本实验)。
- **配对集 = 正典 ne census 的同一批 id 对**(`results/mad/_xcard_cmd_neutral_census_k8.json`,231 matched + 55 ctrl;id=(seed:cluster) 池身份层,与卡文本无关)→ 新脚本 `r7_xcard_v6_census.py` 写 `_xcard_cmd_v6min_census_k8.json`:同 id 对,ccos/n6/n8 换 v6 文本重算。
- **门B**:v6 census 的 id 对集合与 ne census 全等(matched 逐对 + ctrl 集合);ccos 漂移 |Δ| 分布打印(v6 保语义,预期中位 <.02;仅报告不设门槛——TOPIC 匹配质量继承自 ne census,如实注)。
- 双通道**同波**:ne 重导(METHOD=cmd_neutral TAG=r7)+ v6(METHOD=cmd_v6min CARDSRC=data/20mad/cmd_shared_cards_mad__v6min.json TAG=r7),都 DISJOINT=1 全量(231 pos + 231 neg + 55 ctrl = 517 项/通道;比正典 40 对子样功效更足)。正典 .914 只作历史锚,**不做跨波点比**。
- 判官:sonnet 免费子代理 ≤8 并发,亲自逐项判,两通道批次交错派发;pid 覆盖由脚本核对(不信自报)。
- 读数:各通道绝对 AUC + 共享成员簇聚类 CI(cmd_xcard_score 原样)、rare6 基线(v6 预期塌向 .5)、verbatim-free 子集(v6 预期≈全集,本身即结论)、ctrl 阳性对照(share≥2,两通道均值应明显 >.5,<.65 判攻击者弱);**主判决 = 配对 Δ(ne−v6) AUC**,同 id 对、共享成员簇聚类 bootstrap(新脚本 `r7_linkage_paired.py`)。
- 判定:Δ CI-lo>0 → "v6 显著更不可链接";Δ∋0 → "链接主要骑决策架构,消毒不减链接"(如实,与'组合=open'一致);v6 绝对 AUC 无论多少照旧报"非免疫"。预期 v6 落 ~.75–.90(登记直觉,不承诺)。

## 新脚本(全新文件,不动正典)
`scripts/r7_fuzzy_2afc.py`(E1a+E1b)· `scripts/r7_xcard_v6_census.py`(E2 census+门B)· `scripts/r7_linkage_paired.py`(E2 配对差)。

## 门与花费
门A/门B 全过才开判官;门C = 嵌入二跑全缓存命中。COST:嵌入 <$0.05(quote 先行),判官 $0,deepseek $0。

## 必带注(预登记)
- E1b 机械攻击者弱于 LLM 攻击者:绝对匿名认证仍由 LLM 攻击者波持有,E1 只回答"模糊词面通道"这一层。
- E2 同波 ne 参照与正典 .914 是不同波次、不同 n(517 vs 84),只并排展示不做点比。
- v6 的 n6≈0 使 rare6 基线退化(全零分数 AUC=.5)——这是消毒的效果本身,如实报。

## R7 评审修正(sonnet 对抗评审 PASS-WITH-FIXES,6 MAJOR + 8 MINOR,2026-07-17,全采纳)
- **M1 卡内容盲区**(评审实测:忘设 CONSPFC 时六字段门照样全过):门A 增**卡内容层**——E1b 重建后对 neutral/conspf 通道逐 card_id 断言 `p["_card"]` 与 NEUTRALC/CONSPFC 文件的 `k{K}_s{SEED}_{cid}` 键**逐字节相等**;E2 侧新增 **packgate**:由 census + 意图卡文件 + swapped 规则**独立重建每条 batch prompt**,与导出包逐字节比对(sys.txt 亦比对),两通道全过才开判官。
- **M2 import 缓存陷阱**:一个进程只绑一个 METHOD;`r7_xcard_v6_census.py` 的 ne 侧直接 `json.load` 正典 census(不经 XL),v6 侧唯一一次 import XL;顶部 tripwire `assert XL.METHOD == env`。packgate 每包单独进程调用。
- **M3 参照池行数偏置**(评审实测 elements/人 10–44,4.4×):E1a"等规模"钉死为**行数匹配**——成员池与陌生池同构造(k 成员 vs k seeded 陌生人,elements∪aggro,统一切行)后,seeded 子采样两池到 min 行数;两源都保留(与 lex/census 连续,行数匹配后堆叠偏置中和)。
- **M4 档位选择统计**:全局定档一次——每家 indiv 用 card_id 聚类 bootstrap(5000,seed 0,score_2afc_summary 同款)判 SIG(CI-lo>.5);候选档顺序 top5mean→max→mean,取第一个**三家全 SIG**的档;若无,取通过家数最多的最早档,未通过家按 instrument-limited(该家不出 ne/v6 主张)。
- **M5 批次配对**(评审实测两通道批结构逐字节同构):`batch_i(ne)` 与 `batch_i(v6)` **严格同波配对派发**(每波 4 对 = 8 批);`BATCH=24` 显式登记(48 卡结构上限之半);批数/形状以导出打印登记。
- **M6 定档技术隔离**:E1b 拆三段——`MODE=tier` 只算 indiv 三档(不算 ne/conspf)写 per-DS 统计;`MODE=freeze` 汇三家按 M4 规则写 `r7_tier_freeze.json`;`MODE=score` 读冻结档后才算 ne/conspf;单次调用不得同时产出 indiv 与测试通道数字。
- MINOR 采纳:CV 的 env 是 `DATASET=cv`(`results/se/` 只是历史 BATCHDIR 目录名)· ctrl_mean 补未聚类 bootstrap CI,ctrl 排除于 Δ 为有意设计并写明 · 护栏:**paired Δ 永远跑在完整 231 matched 上,verbatim-free 不得限定配对分析** · 全部 recipe 显式 `GROUP=random` + 启动回显 DATASET/K/SEED/GROUP · E1 切行器为 R7 自带统一实现(不 import elemk_v3_gates——它 import 即执行 MODE;卡/elements/aggro/ref 全用同一切行器:newline 切→去 markdown 前缀→≥4 词)· E1b 配对差聚类键 = card_id(与绝对读数同款),E2 配对 Δ 聚类键 = 共享成员 · CARDSRC 是 ROOT 相对路径(≠ NEUTRALC/CONSPFC 裸文件名)。
- 评审已实证免查:三家门A 六字段重建 768/462/696 全过(MAD s0/CV s0/Enron s1);EB.embed 缓存按 (model,text) 哈希确定性;census 55 ctrl + 231 matched 引用的 48 卡 id 全在 v6min 键空间。

### R7 判决登记(2026-07-17,全文见 results/V6R7_FUZZY_LINKAGE_FINDINGS.md)
E1a:v6 杀死模糊词面通道(成员−陌生差 τ≥.90 塌到 +.13/+.09/+.06,τ=.95 ≈0)、保住语义(τ=.70 差 +.58-.69)。
E1b:indiv 阳性对照三家三档全 ns → instrument-limited 分支如实执行,无 ne/v6 推断数字(嵌入内容邻近被 nneg 硬负例对冲)。
E2:同波配对 Δ(ne−v6) 链接 AUC = −.005 [−.054,+.042] ∋0 → **消毒不减跨发布链接**;rare6 基线 .799→.675、
卡-卡 6-gram 6.60→1.89(逐字捷径被砍)但 LLM 链接骑决策架构;v6 绝对 .837 [.800,.873] 照旧高 = 非免疫,
组合=open 不变;任务预期"应下降"被推翻,按预注册分支报。判官 58 批 1034/1034 覆盖;花费 ~$0.02。

---

# R9 预注册(任务 #133):建卡稳定性 —— 新鲜采样重跑 V6 构建(2026-07-18,评审前冻结稿)

## 问题
V6 改写温度 0.3 非确定,正典卡 = 一次随机采样的产物。换一次新鲜采样,证书/路由/丢行/改动量
是否稳定?注意:R6 门A 已证**缓存重放**构建 16/16 卡逐字节复现——那测的是缓存确定性;R9 必须
**绕缓存**取新鲜 API 采样,测的是真随机性下的构建过程稳定性。

## 对象与机制
- 三家正典分区:MAD k8_s0 / CV k8_s0 / Enron k8_s1;输入 = 正典 neutral_fixed 卡 + 同分区成员
  材料(与正典构建逐字节同输入);改写器 = deepseek-chat(正典),prompt/五闸/阈值/重试梯全冻结。
- 绕缓存通道(R12 先例):`v5_sanitize.py` 加 `SAN_SAMPLE` env(int)。设置时全部改写调用改走
  `sample_one(messages, GEN, s=SAN_SAMPLE, temperature=0.3, max_tokens=400)`——缓存键含
  `_sample` → 与正典 chat() 行不同键 → 强制新鲜采样且自身可续跑。不设 = chat() 原路,字节中立。
- TAG=v6min_rerun → 隔离文件 `__v6min_rerun{,_audit,_stats}.json`(audit 同正典 sidecar 访问级,
  永不发布);SAN_SAMPLE=1(0 概念上留给正典)。

## 守卫(防 R6 MAJOR-3 式 $0 假成功)
- assert:SAN_SAMPLE 设置 ⟹ TAG ∉ {v5san, v6min}(否则续跑守卫跳过全部已存簇 = 静默无操作);
- assert:SAN_SAMPLE 设置 ⟹ SAN_EXTRA 为 None(sample_one 不透传 extra;R9 只跑 deepseek);
- `_config` 印章加 SAN_SAMPLE 字段(不同 sample 的续跑必须响);
- 干跑门:ONLY=单簇先跑,断言 ①该簇卡文本 ≠ 正典 v6min 逐字节(新鲜采样证明)②统计过闸,
  然后才放全量三家。

## 判定(预注册,全部程序化,新脚本 `scripts/r9_rebuild_check.py` 不信构建自报)
- **P1 证书复现(硬门)**:`elemk_v3_gates.py MODE=lex CARDS=v6min_rerun` 三家,vs elements 与
  vs aggro 均 **0.0%**。失败 = R9 FAIL 按本名报。
- **P2 丢行门(硬门)**:三家丢行率 ≤5%(正典 3.3/3.5/1.1%;R6 系列已钉成本名指标)。超门 =
  "同模型重采样也可能超门" = 构建不稳定,按本名报;不撤正典。
- **P3 闸独立复检(硬门)**:对每条非丢弃内容行,从文件重算:①数字集保持 nums(orig)⊆nums(new)
  ②punt 状态一致(REQ 正则,冻结 sha 同款)③保真 cos ≥ .65 全局地板(嵌入重算;分档阈值由构建
  执行,此处查全局地板)④行长 ±30%+3 词容差——仅对 strict/relaxed 档(stage-2 rewrite/fallback
  无行长闸,按构建口径)⑤路由确定性:verbatim 行集合与 {rewrite,rewrite_relaxed} 行集合逐一致于
  正典 audit(路由只依赖输入,理应确定)。任一失败 = FAIL。
- **P4 稳定性描述统计(无硬门,如实报)**:tier 分布逐档对照、mean changed_frac、词数比 v6/ne、
  重试率;非 verbatim 行与正典逐字节相同的比例(顺带回答"温度 0.3 有多确定")。
- **P5 反缓存 tripwire(硬门)**:逐卡断言全文 ≠ 正典 v6min 逐字节(多行改写卡在 temp .3 下全字节
  撞车概率≈0;撞了 = 采样没绕开缓存,查 bug 不发数)。

## 效用/匿名不重测的论证(预登记,写进 findings)
- 效用:五闸把语义(cos 地板)、数字、punt 行为、行长全部锁死 → 重跑卡与正典卡在效用仪器敏感的
  全部通道上同分布;R12 已证 FC 认证对大得多的重建扰动(tpar 重抽词面 Jaccard 中位 .25)
  CERT-ROBUST。直接 FC 重测无增量推断力,登记不跑。
- 匿名:改写器同为 deepseek(风格层同分布),词面证书 0% 由 P1 硬验;R1 已证认证对输入侧种子
  变异(扰动大于采样变异)稳健。诚实注:R9 是**构建过程稳定性**主张,不是对重跑卡的新匿名认证。

## 花费与产物
STAGE=cost 实报先行;预估 ~$0.15-0.2(deepseek 重写 ×3 家,费用≈正典构建同量)+ 嵌入 ~$0.01,
判官 $0。超报价 2× 停。产物:`data/{20mad,se,enron}/…__v6min_rerun{,_audit,_stats}.json`、
`results/r9_rebuild_check_{mad,cv,enron}.json`、`results/V6R9_REBUILD_STABILITY_FINDINGS.md`。

## R9 评审修正(sonnet 对抗评审 PASS-WITH-FIXES,4 MAJOR + 6 MINOR,2026-07-18,全采纳)
- **M1(空响应死循环复读缓存)**:重试反馈里唯一不含上轮输出的分支是 empty(FB_EMPTY 静态)→ 连续
  两次空响应后第三轮消息与第二轮逐字节同,sample_one 固定 _sample 下撞键复读缓存坏答案直到耗尽
  (chat() 正典路同病,历史未察)。修:SAN_SAMPLE 分支的 sample_one 调用加 `salt=f"r{rnd}"` 逐轮
  去重(salt 正是 src/llm.py 为此设计的参数);正典 chat() 路不动(字节中立)。
- **M2(P3-⑤ 断言口径)**:只有 {rewrite}∪{rewrite_relaxed} 的**并集**(= route_rw,cov>.90 纯输入
  函数)是确定的;一行落 rewrite 还是 rewrite_relaxed 取决于采样,逐 tier 分别断言会把预期内抖动
  误判 FAIL。修:P3-⑤ = verbatim 集合逐一致 + rewrite 系**并集**逐一致,不比子集。
- **M3("不重测"补否决分支,R12 S1 同款)**:预登记 VETO 门——任一家 |mean_changed_frac(重跑−正典)|
  > .10 或兜底系 tier(relaxed+fallback+fallback_relaxed+rewrite_relaxed)占比偏离 > 15pp(R6
  MINOR-1 同阈)→ 该家"效用不重测"论证作废,须补 confirmatory FC 抽查后才可主张稳定;不触发才可
  沿用论证。
- **M4(env 丢失 = 真实 $0 假成功路径)**:工具环境 shell 状态不跨调用持久 → 全部 env 与 python
  命令必须写在**同一次调用同一行**;干跑门对**三家分别**执行(不是验 MAD 就放 CV/Enron)。
- MINOR 采纳:①TAG 断言只排 {v5san,v6min},对 v6min_qwen 等历史 TAG 靠 _config 印章兜底(SAN_SAMPLE
  字段与已固化印章不等必炸)——此层保护显式写明;②SAN_SAMPLE 判定按 **env 是否存在**(`is not None`),
  显式 "0" 也算设置(防真值判断静默回退);③干跑 ONLY 是子串匹配,必须传完整簇键且避开前缀撞车
  (G1⊂G10…G15;三家统一用 `k{K}_s{SEED}_G2`);④"每簇至少一行非 verbatim"已逐簇实证(39/39),
  换分区需重验;⑤_shingles 与 elemk_v3_gates 逐字节同款但无 sha pin,登记维护风险不动代码;
  ⑥STAGE=cost 三家分别核对,不以首家推断。
- CLEARED:sample_one 不透传 extra ✓;chat/sample_one 键结构不同双向隔离 ✓;新 TAG 文件续跑守卫
  不静默跳 ✓;P3①-④ 口径与 _rewrite_min 实际接受条件逐行核对精确 ✓(行长/数字/punt 仅 stage-1
  接受路径强制,stage-2 无行长闸;全路径 fid 地板 .65);P1 有结构性代码依据 ✓;成本三家合计
  $0.17-0.21 与预估一致、stage_cost ref 指正典 stats 与 TAG 无关 ✓;v6min_rerun 无文件冲突 ✓;
  白名单 .gitignore 下 sidecar 不入库 ✓。

**[R9 干跑修正 2026-07-18]** CV-G2 干跑暴露 P3-⑤ 实现缺陷:丢行 tier="dropped" 掩盖路由归属
(正典 CV-G2 丢 2 行 / 重跑丢 0 行 → audit 标签并集在两侧不可比,M2 版并集比较会误判 FAIL)。
修:P3-⑤ 改为**输入侧重算路由**(_hit_runs + cov>.90,与 _rewrite_min 逐字同式),正典与重跑两侧
audit 各自对照重算真值;丢行豁免标签检查但必须来自非 verbatim 路由。干跑三家 G2 全过:逐字节
差异 ✓(新鲜采样证明)、路由/统计与正典逐档吻合、_config 印章 SAN_SAMPLE=1 ✓。代码评审补充采纳:
SAN_SAMPLE⟹GEN=deepseek 显式断言、changed_frac 空表 NaN 防御、ensure_ascii=False。
三家报价实测:MAD $0.08 / CV $0.04 / Enron $0.07 ≈ $0.19 合计(包络内)。

**[R9 干跑修正 2 2026-07-18]** 全量后 CV/Enron 核查在 extract 逆变换崩(IndexError):
`"\n".join(out_lines).splitlines()` 往返恰好吃掉**一个尾部空元素**——当卡的最后一条内容行被丢弃、
卡以空行收尾时(CV G4),文件比幸存行数短一行。这是构建器既有装配语义(化妆性,正典卡同样适用),
非 R9 引入。修:extract 容忍"唯一剩余幸存行 = 尾部空的非内容行"这一模式,其余不匹配照旧 FAIL。

# R10 登记(任务 #134):第三梯队论证收尾 + 文档换头(2026-07-18,$0)

① **worst-case 逐字暴露 on v6 = 每个成员 0(推论,无需新跑)**:词面证书(elemk_v3_gates MODE=lex,
独立于构建测量 ×3 家)按簇对"全体成员 elements∪aggro 的并集"查 6 词连串,0.0% ⇒ 不存在任何一行与
**任何一个成员**共享 ≥6 词连串 ⇒ 逐人 worst-case 逐字暴露轴(退化时代曾达 ~700 词整段)对 v6 收口
清零。构造+测量双重保证,登记为推论。
② **A4 个体尾巴转移论证(登记为论证,非测量)**:ne 时代 A4 = 均值 ~.575 背后 6-17% 个体 k8 单发布
可稳定再识别。v6 不重测;论证 = 配对差 ne−v6 ∋0 ×3(两攻击者谱系)+ R13 MAD 全 k 认证(ne 低 k
尾巴的驱动被消毒拆掉)+ R4 Enron 残余=弥散输入侧非个体尖峰。诚实注:v6 逐人尾巴未直测,论文按
"论证"措辞,不升级为测量主张。
③ **文档换头(CMD := pool + V6,ne 降级为未消毒基线)**:README.md(方法节四步化+V6 正典判决块+
旧 Key results 加 pre-V6 横幅+repo layout 补 v5_sanitize/elemk_v3_gates/r9_rebuild_check)·
story_line.md(★★ 换头块+一行 spine 更新+★ 节降级为池化层)· story_line.html(顶部换头横幅)·
data_provenance.md(★★ 2026-07-18 正典块,A′/C′ 标为 baseline layer,R10 推论登记)·
memory benchmark-plan/elemk 同步。组合/多发布照旧 open direction 措辞。
**V6 替代阶梯(R1–R13)至此全部完成;下一步 = #56 论文正文(单一底稿 results/PAPER_EVIDENCE_PACK.md)。**

### R9 判决登记(2026-07-18,全文见 results/V6R9_REBUILD_STABILITY_FINDINGS.md)
**STABLE**:P1 词面证书 0.0%×3 复现(vs elements 与 vs aggro 双参照);P2 丢行 2.5/3.5/2.8%
(正典 3.3/3.5/1.1%)全 ≤5% 门;P3 五闸独立复检 518/219/418 改动行全绿 + verbatim 字节等 +
路由确定性输入侧重算实证(verbatim 73/29/38 逐一致);P5 反缓存逐卡字节差异 39/39;
changed_frac .450→.443 / .417→.426 / .511→.495、兜底系占比漂移 ≤4pp → **M3 VETO 未触发,
"效用/匿名不重测"论证生效**。改动行与正典逐字同仅 8.8–12.7% = 真新鲜采样。机制表述:
"改哪里"是输入的确定函数,"怎么改"随机但被五闸夹死。花费 ~$0.19,判官 $0。
