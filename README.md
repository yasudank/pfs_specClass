# pfs_specClass

開発の現状と再開手順は `CLAUDE.md`、経緯は `docs/dev_log.md`。

Subaru/PFS スペクトル（pfsObject coadd）の教師なし分類・異常スペクトル検出。

## 構成

```
src/pfs_specclass/
  preprocess.py   pfsObject FITS 読み込み → 共通 log-λ グリッドへ再ビン（マスク・分散・m アーム平滑化）
  dataset.py      キューブ読み込み、規格化、有効重み（sky / flux floor / アーム端トリム）、天体単位の fold
  wpca.py         欠損・重み付き PCA（GPU, 交互重み付き最小二乗）
  model.py        spender 型オートエンコーダ（χ² 損失、振幅解析解、赤方偏移対応デコーダ）
  quality.py      品質指標・ホットファイバー判定・品質フラグ・重み調整
  visit_quality.py  露出・カメラ単位の較正異常（ホット visit）の判定
  scoring.py      学習済みモデルで新スペクトルを潜在化・クラスタ・kNN ラベル・異常度・品質フラグ
  plots.py        診断図
scripts/
  build_dataset.py   全 FITS → data/pfs_spectra.h5 + _meta.parquet（DB targets と結合）
  build_quality.py   品質指標とフラグ → data/pfs_spectra_quality.parquet, _hot_fibers.csv
  build_visit_quality.py  ホット visit → _visit_units.parquet、quality.parquet に HOT_VISIT と hot_bins を追加
  run_wpca.py        PCA ベースライン → outputs/wpca/
  train_ae.py        AE 学習 + GMM 密度 + UMAP + 異常ギャラリー → outputs/ae/
  score_spectra.py   新しい FITS をスコアリング
```

## 使い方

```bash
uv sync
uv run python scripts/build_dataset.py --out data/pfs_spectra.h5      # 約 3〜7 分
uv run python scripts/build_quality.py                                # 約 1 分
uv run python scripts/build_visit_quality.py --model outputs/ae_v4    # 約 3 分（build_quality の後に実行）
uv run python scripts/run_wpca.py --out outputs/wpca                  # 約 2 分
uv run python scripts/train_ae.py --out outputs/ae_v7 --exclude NEG_BLUE,NEG_NIR  # 約 65 分 (GB10, 50 epoch)
uv run python scripts/compare.py --ae outputs/ae_v7/ae_results.parquet # 表現の比較
uv run python scripts/score_spectra.py --model outputs/ae_v7_s1 --out scores.csv path/to/pfsObject*.fits
```

## データ設計

- **スペクトルの一意キー**: `(catId, objId, combination)`。objId はファイル名ではなく TARGET HDU の
  header から取る（ファイル名の数値は丸められている場合がある）。
- **学習/検証分割の単位**: 天体 `(catId, objId)`。`fold = crc32("catId_objId") % 10`。
  fold 0–7 学習、8 検証、9 テスト。同一天体の別 combination は必ず同じ fold。
- **波長グリッド**: 376–1268 nm、Δlnλ = 3.5e-4（約 105 km/s）、3474 ビン。combination ごとに
  元グリッドが異なる（11251 / 11501 / 13293 px）ため共通化。
- **マスク**: MASK のいずれかのビットが立つピクセルは重み 0（`read_pfsobject(mask_bits=...)` で変更可）。
  さらに各連続区間の両端 3 ビン、10 ビン以上の内部欠損（bmn の 656–703 / 896–937 nm など）の
  両側 25 ビン（約 5〜8 nm）を除外（アーム端の大きなスパイク対策）。
- **m アーム**（bmn_*, selected_run24）: 700–890 nm を r アーム相当分解能（R≈3000）へガウス平滑化。
- **有効分散**: `var = 1/ivar + (sky_frac·sky)^2 + (flux_frac·|f|)^2`（規格化後）。
  sky_frac=0.04 は PCA 残差から較正（大気輝線の明るさに比例する残差がおよそ 3〜4%）。flux_frac=0.02 は
  高 S/N スペクトルが学習を支配しないための床。
- **規格化**: 有効ビンのフラックス中央値で割る（`norm`）。AE はさらに振幅を解析的に解く。

## 異常度

`anom_chi2`（再構成 χ²/ν）と `anom_density`（潜在空間 GMM の −log p）を、**S/N 十分位ビン内**の
パーセンタイルに変換（低 S/N が単にノイズで上位に来ないように）。`anom_score = max(両者)`。

## アーム構成への不変性

b+m+n の coadd は 656–703 nm と 896–937 nm にデータがない。何も対策しないと、エンコーダがこの
欠損パターン（＝アーム構成）を学習してしまう。学習時は、エンコーダへの入力に (a) ランダムな
ブロックマスクと (b) 確率 0.5 で bmn と同じ欠損区間のマスクをかけ、異なる拡張をかけた 2 つの入力の
潜在ベクトルが一致するよう一貫性損失（`--consistency`）を加える。

## 結果（2026-10-06, 保留天体 fold 8–9, S/N>5）

| 指標 | wPCA (k=10) | AE v1 | AE v2 | AE v3（品質フラグ） | **AE v4**（+ MIXED_RM） |
|---|---|---|---|---|---|
| 再構成 χ²/ν 中央値 | 1.545 | 1.786 | 1.539 | 1.477※ | 1.470※ |
| 同一天体の別 combination の潜在距離 / ランダムペアの距離 | 0.495 | 0.431 | 0.268 | 0.287 | 0.255 |
| kNN による DRP 分類の再現率（全体 / QSO） | 0.910 / 0.456 | 0.930 / 0.500 | 0.939 / 0.506 | 0.938 / 0.500 | 0.938 / 0.570 |
| m アーム有無の予測 AUC（logistic、0.5 が理想） | 0.821 | 0.973 | 0.871 | 0.908 | 0.861 |
| 低密度側の上位 100 本に含まれる HOT_FIBER / n アームを落としたスペクトル | – | – | 24 / 10 | 4 / 3 | – |
| 低密度側の上位 100 本に含まれる NEG_BLUE | – | – | 9 | 64 | – |
| 低密度側の上位 100 本のうち欠陥フラグ付き（MIXED_RM は除く） | – | – | 0.55 | 0.85 | 0.72 |
| bmnr の箱型のずれ 101 本: χ²/ν 中央値 / 異常度 0.99 超の割合 | – | – | – | 3.54 / 26.7% | 2.61 / 12.9% |

※ v3 はマスクした区間を評価に含めないので、χ² は v2 とそのまま比べられない。

v3 では、n アームの装置由来のスペクトルが異常度の上位から大きく減った。NEG_BLUE が増えたのは、
学習から除外した壊れたスペクトルが潜在空間の分布の外に出るため（意図どおり）。品質フラグのない
スペクトルだけで並べた一覧は `top_*_clean.png`。

v1 から v2 での変更点: InstanceNorm を外し、粗くビン化した連続光（32 区間）を MLP に直接入力、
bmn 欠損区間のデータ拡張と一貫性損失を追加。

m アームの AUC が 0.5 にならない主な原因は、観測した天体集団の違い（combination ごとに視野が異なる）
と考えられる。同じ天体で比べると、アーム構成が異なるペアと同じペアの潜在距離はほぼ等しい
（0.900 と 0.887）。

`outputs/ae/` は v1 の実験結果で、現在のコード（Encoder の構造を変更）では読み込めない。
`outputs/ae_v2/` は品質フラグなし（`--no-quality` 相当）、`outputs/ae_v3/` は MIXED_RM なし。
v4 / v6 / v7 の比較と、推奨モデルの扱い（判断保留中）は CLAUDE.md と docs/dev_log.md を参照。

### つなぎ目の段差の調査（2026-10-07）

AE v3 の再構成を基準に、つなぎ目の両側の相対残差の差を段差の指標として測った（S/N>5、
`outputs/diagnostics/junction_step_samples.png`）。

| 段差 | 15% 超かつ 5σ 以上の割合 | 判断 |
|---|---|---|
| bmnr の r だけの区間（656–703, 896–937 nm）の箱型のずれ | bmnr の 2.2% / 4.1%（bnr で同じ区間を測ると 0.15% / 0.74%） | 装置由来。MIXED_RM として全 bmnr でこの区間を落とす |
| b/r のつなぎ目（660–690 vs 610–640 nm、bnr） | 1.2% | フラグにしない。立つ例の多くは 650 nm 付近に輝線のある QSO や、再構成の合わない強い TiO の M 型星 |
| r/n のつなぎ目（975–1010 vs 905–935 nm、bnr） | 4.5% | 950 nm の 15〜20% の段差が見える例が多いが QSO なども混ざるため、今はフラグにしない |

以前「b/r のつなぎ目の段差」と書いた例（selected_run25）の実体は、bmnr の箱型のずれだった。
例えば objId 110531500621946746 は、bmn_run26 / brn_run26 では DRP が STAR とするのに、
selected_run25（bmnr）では箱型のずれのために GALAXY と判定されている。

## 品質フラグ（quality.py）

| フラグ | 定義 | 重みの扱い |
|---|---|---|
| NARM_BUMP | peak = f(975–1005) / mean(f(915–935), f(1030–1060)) > 1.2 | ホットファイバー上: n アーム (>925 nm) 全体を 0。それ以外: 925–1015 nm を 0 |
| NARM_STEP | rise = f(1030–1060) / f(915–935) > 1.5 | ホットファイバー上のみ n アーム全体を 0（それ以外は本物の赤い天体が多いので何もしない） |
| HOT_FIBER | いずれかの visit でホットファイバーを使用 | （上記の判定に使用） |
| NEG_BLUE | 380–480 nm の平均 < −0.05（規格化後）かつ 5σ 以上 | 学習から除外 |
| NEG_NIR | 1150–1260 nm の平均 < −0.05 かつ 5σ 以上 | 学習から除外 |
| MIXED_RM | m アームと r アームの visit を両方含む coadd（arms=bmnr） | 650–708.5 nm と 888–945.5 nm を 0（「扱いの印」で欠陥扱いはしない） |
| HOT_VISIT | 較正異常のある露出・カメラ（下記）を含む | そのスペクトルのホットな 20 nm ビンを 0 |
| HOT_VISIT_HEAVY | HOT_VISIT のうちホットなビンが 3 つ以上 | 学習から除外 |

**学習データ（v6 以降）**: NEG_BLUE、NEG_NIR、HOT_FIBER、NARM_BUMP、HOT_VISIT_HEAVY のいずれかが立つスペクトルは
学習に使わない（S/N>5 の 12.2%）。NARM_STEP だけのもの（本物の赤い星が多い）、MIXED_RM、ビンの少ない HOT_VISIT は
重みの調整だけで使う。天体はファイバーにほぼランダムに割り当てられるので、ファイバー単位の除外は天体集団を
偏らせないが、visit は視野・run・アーム構成と結びついているので、visit 単位の除外は集団を偏らせる。そのため
ホット visit は影響の大きいスペクトルだけを除外する。除外の割合は combination ごとに 4〜10%、selected_run24 で 25%、
selected_S25A（S/N>5 はわずか）で 100%。重みの調整は学習・評価・推論のすべてで行う。

v5（HOT_VISIT をすべて除外、学習 67,158 本）は、HOT_VISIT の判定がアーム構成の違いを拾っていた（bmn_run26 の 71% が
HOT_VISIT）ため 10 エポックで中止した（`outputs/ae_v5_aborted/`）。

### ホット visit（visit_quality.py）

1. 各スペクトルについて、アーム（b: 380–645、red: 650–940、n: 945–1260 nm）を 1 本ずつエンコーダから隠し、
   他のアームから予測した再構成との相対残差を 20 nm ビン（45 個）で求める。隠したアームの装置由来の形は
   モデルが再現できないので、装置由来の形を学習してしまったモデルでも使える。
2. 予測の偏りは天体の種類・赤化・アーム構成に依存するので、同じアーム構成（m を含むか否か）で、別の visit の組の
   近傍 40 本の残差の中央値を差し引く。近傍は**そのアームを隠してエンコードした潜在ベクトル**で探す（全体から
   エンコードした潜在ベクトルは装置由来の形そのものを含み、同じ形を持つ近傍を選んで打ち消してしまうため）。
3. 「visit の組 × 分光器」と「1 つの visit × 分光器」の 2 種類の単位で中央値を取り、(a) 誤差の 4 倍以上、
   (b) 5% 以上、(c) 同じアーム構成の単位間のばらつき（ロバスト標準偏差）の 4 倍以上ずれているビンをホットとする
   （単位あたり 20 本以上、S/N>3）。前者は感度が高く、後者は全体の 99% をカバーする。
4. スペクトルは、属するすべての単位のホットなビンの和集合を `hot_bins`（int64 のビットマスク）として持つ。

現在、ホットな単位は visit の組 × 分光器で 140 / 1139、visit × 分光器で 873 / 4442。HOT_VISIT は全スペクトルの
10.5%（ビン数の中央値 5）、HOT_VISIT_HEAVY は S/N>5 の 5.2%。UMAP の「島 1」（visit 134908–134915 の n アームの
異常）の 78% を HOT_VISIT として検出する（閾値 5σ・5 倍では 39%、3 倍にすると 89% だが HOT_VISIT が 17% に増える）。
推論では、学習時の単位の表（モデルのディレクトリの `visit_units.parquet`）に載っている visit だけ判定できる。

ホットファイバーは、BUMP または STEP の発生率を全体の発生率と比べる二項検定で判定する
（単独ファイバーで p<1e-4 かつ 3 倍以上、または隣接 ±3 本をまとめたときに同じ条件を満たし、
かつ単独でも 2 倍以上）。現在は、各分光器の高 fiberId 側の端（610–671, 1274–1301, 1937–1952）と
1132–1151 付近など 114 本（`outputs/diagnostics/hot_fibers.png`）。モデルのディレクトリに
`hot_fibers.csv` をコピーするので、推論時も同じ判定を使う。

本物の幅の広い輝線が 925–1015 nm にある天体（例: z≈0.5 の Hα、z≈1 の Hβ を持つ QSO）は、NARM_BUMP で
その区間がマスクされる可能性がある。フラグは出力に残るので、別ファイバーでの再観測があれば
それで確認する。
