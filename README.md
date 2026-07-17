# openduck_operator_side

このリポジトリは、[Open_Duck_Playground](https://github.com/fabc-why/Open_Duck_Playground) から、ROS2 経由で OpenDuck を操作する部分を切り分けたものです。

今後はいろいろなライブラリを追加していく可能性があるため、依存関係の競合や運用上のリスクを抑え、管理しやすい構成にすることを目的としています。

## ローカル LLM 操作

`scripts/auto/llm/local_llm_1.py` は、カメラ画像とテキストの指示を Ollama に渡して、返答を文字列アクションとして ROS2 経由で OpenDuck に反映する構成です。

プロンプトインジェクション対策として、画像やテキスト内の指示は untrusted として扱い、モデルには固定の制御方針と JSON 形式の出力だけを許可しています。

起動時に `--task` で操作目的を渡せます。例:

```bash
python3 scripts/auto/llm/local_llm_1.py --task "赤い物体の方向へ安全に近づいてください"
```
