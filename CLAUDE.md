# CLAUDE.md — pfs_specClass 作業の引き継ぎ

Subaru/PFS の pfsObject スペクトルを教師なし学習で分類し、(a) 新しいスペクトルの分類器と (b) 特殊なスペクトルの
検出の基礎を作るプロジェクト。2026-10-06〜07 に GB10（aarch64, CUDA 13）のマシンで開発し、以後 RTX 4090 の
マシンで続ける。手法・ファイル構成・品質フラグの定義は `README.md`、経緯と判断の理由は `docs/dev_log.md` にある。

## ユーザーについて

- PFS の天文学者。**日本語でやり取りする。**
- データの同一性や装置の話には正確さを求める（例: 1 本のスペクトルを一意に決めるのは (catId, objId) ではなく
  (catId, objId, combination)）。キー・単位・アーム構成などを曖昧に書かない。
- 推測と確認済みの事実を区別して報告する。学習結果は数値と図で示し、悪化した指標も隠さない。
- 1 時間程度の学習は了承を得てから実行してきた。長い学習は `setsid nohup` で切り離して起動する（下記）。

## データの場所（このディレクトリの外、読み取り専用として扱う）

- FITS: `/mnt/ugnas/work/PFS/CommFiller/pfs-target-viewer/extracted_targets/fits`（216,476 本）
- DB: `/mnt/ugnas/work/PFS/CommFiller/pfs-target-viewer/pfs_metadata.sqlite3`（`targets` 表と結合）
- 新しいマシンでもこのパスでマウントされているか最初に確認する。違う場合は `scripts/build_dataset.py` の
  `DEFAULT_FITS_ROOT` / `DEFAULT_DB` を引数で指定する（キューブは作成済みなので、通常は再構築不要）。

## 生成済みのもの（`data/` 8.6 GB、`outputs/` 0.6 GB。.gitignore 対象）

| ファイル | 内容 |
|---|---|
| `data/pfs_spectra.h5` | 共通 log-λ グリッド（3474 ビン）の flux / ivar / sky キューブ |
| `data/pfs_spectra_meta.parquet` | 1 行 1 スペクトル（行番号 `row` がキューブの行）。DB の targets と結合済み |
| `data/pfs_spectra_quality.parquet` | 品質指標・`qflag`・`hot_bins`（build_quality → build_visit_quality の順で作成） |
| `data/pfs_spectra_hot_fibers.csv` | ファイバーごとの n アーム異常の統計とホット判定 |
| `data/pfs_spectra_visit_units.parquet` | 露出・カメラ単位（visit の組 × SM、visit × SM）のずれとホットなビン |
| `data/pfs_spectra_armres.npz` | leave-one-arm-out 残差 R とその偏差 D（診断用） |
| `outputs/ae_v*/` | 各モデル（`ae.pt`、`gmm.joblib`、`ae_results.parquet`、UMAP・異常ギャラリー） |
| `outputs/diagnostics/` | 調査に使った図と表 |
| `outputs/compare_v4_v6_v7.txt` | 最新の比較表 |

## 新しいマシンでの再開手順

**重要: `.venv/` は共有ディレクトリ内にあり GB10（aarch64）用。新しいマシンで `uv sync` をそのまま実行すると
これを上書きして GB10 側の環境を壊す。** 仮想環境は必ずマシンごとに別の場所に作る。

```bash
cd /mnt/ugnas/work/PFS/pfs_specClass
nvidia-smi                       # ドライバ 580 以上なら cu130 版 PyTorch が使える
export UV_PROJECT_ENVIRONMENT=$HOME/.venvs/pfs_specclass   # 4090 マシン専用の仮想環境（~/.bashrc に書いておく）
uv sync                           # uv.lock に x86_64 + cu130 の torch 2.14.1 が入っている
uv run python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
```

- ドライバが古く cu130 が使えない場合は、`pyproject.toml` の `pytorch-cu130` を `cu128` に変えて
  `uv lock` し直す（GB10 側にも影響するので、その旨をユーザーに伝える）。
- メインメモリ: `train_ae.py` は全キューブを規格化して読み込むため**ピーク約 28 GiB** 必要。足りなければ
  分割読み込みに直す。GPU メモリはピーク約 7.7 GiB（バッチ 256）なので 24 GB で余裕がある。
- 速度: GB10 では 1 エポック約 72 秒（50 エポック + 評価で約 65 分）。4090 では仕様上 2〜4 倍速い見込み（未計測）。
  最初に 1〜2 エポックで速度を測るとよい（`--epochs 2 --out outputs/tmp_speed`）。
- キューブを NAS から読むと遅い場合は、`data/` をローカルディスクにコピーして `--data` で指定する。

## よく使うコマンド

```bash
uv run python scripts/build_quality.py                               # 品質フラグ（約 1 分）
uv run python scripts/build_visit_quality.py --model outputs/ae_v4   # ホット visit（約 3 分、build_quality の後）
uv run python scripts/train_ae.py --out outputs/ae_vN --exclude NEG_BLUE,NEG_NIR --seed 0
uv run python scripts/compare.py --ae outputs/ae_v4/ae_results.parquet outputs/ae_vN/ae_results.parquet
uv run python scripts/score_spectra.py --model outputs/ae_v7_s1 --out scores.csv FILE.fits ...
```

長い学習は Claude Code から切り離して起動する（セッションを閉じても続く。途中保存はないので、マシンが
止まったら最初からやり直し）:

```bash
setsid nohup scripts/run_v7.sh > outputs/run_v7.out 2>&1 < /dev/null &   # run_v7.sh を雛形にする
```

## 現在の状況（2026-10-07）

| モデル | 学習から除外 | 重みの調整 | 備考 |
|---|---|---|---|
| ae_v2 | なし | なし | 品質フラグ導入前 |
| ae_v3 | NEG_* | n アーム関連 | |
| ae_v4 | NEG_* | + MIXED_RM | `build_visit_quality.py` の既定モデル（hot_bins はこのモデルで計算） |
| ae_v5_aborted | + HOT_VISIT 全部 | | HOT_VISIT がアーム構成の違いを拾っていたため中止 |
| ae_v6 | NEG_*, HOT_FIBER, NARM_BUMP, HOT_VISIT_HEAVY | + HOT_VISIT のビン | `EXCLUDE_FROM_TRAINING` の既定値 |
| ae_v7, ae_v7_s1 | NEG_* | + HOT_VISIT のビン | シード 0 / 1 |

欠陥フラグのないスペクトル（評価用 fold、S/N>5）での比較:

| 指標 | v4 | v6 | v7 | v7_s1 |
|---|---|---|---|---|
| 再構成 χ²/ν 中央値 | 1.492 | 1.551 | 1.516 | 1.463 |
| ホットファイバー AUC（0.5 が理想） | 0.692 | 0.700 | 0.727 | 0.701 |
| 分光器 AUC（0.5 が理想） | 0.637 | 0.618 | 0.657 | 0.627 |
| m アーム AUC（0.5 が理想） | 0.870 | 0.852 | 0.898 | 0.838 |
| kNN の DRP 分類再現（全体 / QSO, n=131） | 0.941 / 0.588 | 0.938 / 0.466 | 0.940 / 0.466 | 0.941 / 0.496 |

**結論**: シードだけで指標が 0.03〜0.06 動き、学習データの選び方による差はその範囲に収まる。除外を強める利点は
見えないので、データを捨てずに重みで装置由来の部分を落とす v7 の構成を既定にする方針を提案済み。ただし
**ユーザーの最終判断はまだ**（既定モデルの切り替え、README の「推奨モデル」の更新は保留）。
`scripts/score_spectra.py` の `--model` の既定値は現在 `outputs/ae_v4`。

## 次の作業の候補（ユーザーの指示待ち）

1. 既定モデルを ae_v7_s1（v7 の構成）にするかの決定。
2. 装置情報を潜在空間から積極的に除く構造の改良（ホットファイバー AUC が約 0.70 から下がらないため）:
   - デコーダに装置の情報（分光器、端のファイバーか、アーム構成）を条件として与える
   - 潜在ベクトルから分光器・ホットファイバーを当てる分類器に対する敵対的損失
   どちらも複数シードで比較する（シードの揺らぎが大きいため）。
3. その他の残課題: r/n のつなぎ目の段差（950 nm, bnr の 4.5%）のフラグ化、静止系での銀河の学習
   （DRP の z は z≈0.88 に偽の集中があるので注意）、UMAP の島 2（selected_run24 の晩期 M 型星らしき集団）の確認、
   ホット visit の判定の改良（島 1 の 78% を検出）、normalizing flow による密度推定。

## 実装上の注意

- objId はファイル名でなく FITS の TARGET header から取る（ファイル名の数値は丸められていることがある）。
- 学習・検証の分割は天体単位: `fold = crc32("catId_objId") % 10`（0–7 学習、8 検証、9 テスト）。
- `qflag` はビットマスク。MIXED_RM は「扱いの印」で欠陥扱いしない（`quality.is_clean`）。
- `build_quality.py` は qflag を作り直すので、その後に必ず `build_visit_quality.py` を実行する。
- 比較は `scripts/compare.py`。学習から除外したスペクトルは分布の外に出るので、モデル間の公平な比較には
  欠陥フラグのないスペクトルに限った指標も見る（`docs/dev_log.md` 参照）。
- シェルで `... | grep -v ...` の出力が空だと終了コード 1 になる（エラーではない）。
- 作図は `plots.py` の配色（カテゴリ 3 色まで、連続量は青の単色ランプ）に合わせる。
- git は初期化済みだがコミットはまだない。
