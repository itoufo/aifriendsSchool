# クイズJSONファイル仕様

## ファイル命名規則
- ファイル名は章のIDと一致させる: `{docId}.json`
- 例: `ai-literacy.json`, `prompt-basics.json`

## JSONスキーマ

```json
{
  "docId": "章のID（必須）",
  "title": "クイズのタイトル",
  "questions": [
    {
      "id": "質問の一意ID",
      "type": "single | multiple",  // single: 単一選択, multiple: 複数選択
      "question": "質問文",
      "options": [
        {
          "id": "選択肢ID",
          "text": "選択肢のテキスト",
          "isCorrect": true/false
        }
      ],
      "explanation": "解説文（オプション）"
    }
  ]
}
```

## サンプルファイル
- `sample-quiz.json`: 基本的なクイズのサンプル
- 各章のIDに対応するJSONファイルを作成してください

## 注意事項
- UTF-8エンコーディングを使用
- 有効なJSONフォーマットであることを確認
- 少なくとも1つ以上の正解選択肢を含める