import type { Quiz } from './quiz.types';

/**
 * 問題データ
 * 各章に対応する復習問題を定義
 * MDファイルの確認クイズから抽出
 */
export const quizzes: Quiz[] = [
  // ========== 第1段階：入門 ==========
  {
    id: 'quiz-ai-literacy',
    chapterId: 'ai-literacy',
    title: 'AIリテラシー入門 確認クイズ',
    description: 'AIの基本概念に関する理解度を確認しましょう。',
    questions: [
      {
        id: 'q1',
        type: 'single',
        question: '次のうち、AIの説明として最も適切なものはどれですか？',
        options: [
          { id: 'a', text: '人間の脳を完全に再現したコンピュータ', isCorrect: false },
          { id: 'b', text: '人間の知的活動をコンピュータで実現する技術', isCorrect: true },
          { id: 'c', text: '自我を持ったコンピュータプログラム', isCorrect: false },
          { id: 'd', text: 'インターネットに接続されたすべてのコンピュータ', isCorrect: false },
        ],
        explanation: 'AI（人工知能）とは、人間の知的活動（学習、推論、認識、判断など）をコンピュータで実現する技術の総称です。',
      },
      {
        id: 'q2',
        type: 'single',
        question: '次のうち、現在のAIが苦手とすることはどれですか？',
        options: [
          { id: 'a', text: '大量のデータからパターンを見つける', isCorrect: false },
          { id: 'b', text: '画像の中から特定の物体を認識する', isCorrect: false },
          { id: 'c', text: '学習データにない状況での倫理的判断', isCorrect: true },
          { id: 'd', text: '定型的な文章を素早く生成する', isCorrect: false },
        ],
        explanation: 'AIはパターン認識や定型的なタスクは得意ですが、学習データにない状況での倫理的判断や創造的判断は苦手とします。',
      },
      {
        id: 'q3',
        type: 'single',
        question: '大規模言語モデル（LLM）の動作原理として正しいものはどれですか？',
        options: [
          { id: 'a', text: '人間のように「考えて」回答を生成している', isCorrect: false },
          { id: 'b', text: 'あらかじめ用意された回答データベースから検索している', isCorrect: false },
          { id: 'c', text: '学習データのパターンを基に確率的に次の単語を予測している', isCorrect: true },
          { id: 'd', text: 'インターネットをリアルタイムで検索して回答している', isCorrect: false },
        ],
        explanation: 'LLMは「理解」しているように見えますが、実際は学習データのパターンを基に統計的に次の単語を予測しています。',
      },
    ],
  },
  {
    id: 'quiz-generative-ai',
    chapterId: 'generative-ai',
    title: '生成AIの仕組み 確認クイズ',
    description: '生成AIとLLMの基本的な仕組みを理解しましょう。',
    questions: [
      {
        id: 'q1',
        type: 'single',
        question: 'Transformerアーキテクチャの核心となる技術は何ですか？',
        options: [
          { id: 'a', text: '畳み込み処理', isCorrect: false },
          { id: 'b', text: '自己注意機構（Self-Attention）', isCorrect: true },
          { id: 'c', text: '再帰的処理', isCorrect: false },
          { id: 'd', text: 'ルールベース処理', isCorrect: false },
        ],
        explanation: 'Transformerの核心は自己注意機構（Self-Attention）で、入力シーケンス内の各要素間の関連性を計算します。',
      },
      {
        id: 'q2',
        type: 'single',
        question: 'LLMの事前学習で主に行われるタスクは何ですか？',
        options: [
          { id: 'a', text: '質問に正確に回答する', isCorrect: false },
          { id: 'b', text: '次の単語を予測する', isCorrect: true },
          { id: 'c', text: '文法の正誤を判定する', isCorrect: false },
          { id: 'd', text: '感情を分析する', isCorrect: false },
        ],
        explanation: 'LLMは主に「次の単語を予測する」タスクで大量のテキストから学習します。',
      },
      {
        id: 'q3',
        type: 'single',
        question: 'ハルシネーションが起きる主な原因として正しいものは？',
        options: [
          { id: 'a', text: 'AIが意図的に嘘をついている', isCorrect: false },
          { id: 'b', text: 'インターネット接続が不安定', isCorrect: false },
          { id: 'c', text: 'もっともらしい文章を生成するよう学習しており、事実の正確性は直接学習していない', isCorrect: true },
          { id: 'd', text: 'ユーザーの質問が曖昧だから', isCorrect: false },
        ],
        explanation: 'ハルシネーションはAIが「もっともらしい」文章を生成するよう学習しているため、事実と異なる内容を自信を持って生成してしまう現象です。',
      },
    ],
  },
  {
    id: 'quiz-ai-ethics',
    chapterId: 'ai-ethics',
    title: 'AI倫理と安全性 確認クイズ',
    description: 'AI活用における倫理と安全性を確認しましょう。',
    questions: [
      {
        id: 'q1',
        type: 'single',
        question: 'AI生成物の著作権について正しいものはどれですか？',
        options: [
          { id: 'a', text: 'AIが生成したものにはすべて著作権が発生する', isCorrect: false },
          { id: 'b', text: 'AIが生成したものには一切著作権が発生しない', isCorrect: false },
          { id: 'c', text: '人間の創作的関与の度合いによって判断が分かれる', isCorrect: true },
          { id: 'd', text: 'AIサービス提供会社に著作権が帰属する', isCorrect: false },
        ],
        explanation: 'AI生成物の著作権は、人間がどの程度創作的に関与したかによって判断が分かれます。',
      },
      {
        id: 'q2',
        type: 'single',
        question: '生成AIに個人情報を入力する際の対策として不適切なものは？',
        options: [
          { id: 'a', text: '氏名を匿名化してから入力する', isCorrect: false },
          { id: 'b', text: '企業向けプラン（学習に使われない）を利用する', isCorrect: false },
          { id: 'c', text: '重要でない個人情報ならそのまま入力して良い', isCorrect: true },
          { id: 'd', text: '入力前に個人情報を削除する', isCorrect: false },
        ],
        explanation: '「重要でない」個人情報という判断は危険です。すべての個人情報は適切に保護する必要があります。',
      },
      {
        id: 'q3',
        type: 'single',
        question: 'AIバイアスへの対処として適切なものは？',
        options: [
          { id: 'a', text: 'AIの出力は常に中立なので対処不要', isCorrect: false },
          { id: 'b', text: 'バイアスの存在を認識し、重要な判断は人間が行う', isCorrect: true },
          { id: 'c', text: 'バイアスは技術的に解決済みなので心配不要', isCorrect: false },
          { id: 'd', text: 'バイアスを避けるためAIの利用を中止する', isCorrect: false },
        ],
        explanation: 'AIにはバイアスが存在することを認識し、重要な判断は人間が行うことが適切です。',
      },
    ],
  },
  {
    id: 'quiz-prompt-basics',
    chapterId: 'prompt-basics',
    title: 'プロンプト基礎 確認クイズ',
    description: '効果的なプロンプトの書き方を確認しましょう。',
    questions: [
      {
        id: 'q1',
        type: 'single',
        question: '効果的なプロンプトに含めるべき4要素として正しい組み合わせは？',
        options: [
          { id: 'a', text: '目的、文脈、出力形式、フォーマット', isCorrect: true },
          { id: 'b', text: '主語、述語、目的語、修飾語', isCorrect: false },
          { id: 'c', text: '導入、本論、結論、まとめ', isCorrect: false },
          { id: 'd', text: '質問、回答、確認、完了', isCorrect: false },
        ],
        explanation: 'プロンプトの4要素は、Goal（目的）、Context（文脈）、Output（出力形式）、Format（フォーマット）です。',
      },
      {
        id: 'q2',
        type: 'single',
        question: 'Few-shotプロンプトの説明として正しいものは？',
        options: [
          { id: 'a', text: '短い文章でプロンプトを書く手法', isCorrect: false },
          { id: 'b', text: '少数の例を示してAIに学習させる手法', isCorrect: true },
          { id: 'c', text: '複数のAIを同時に使う手法', isCorrect: false },
          { id: 'd', text: '画像とテキストを組み合わせる手法', isCorrect: false },
        ],
        explanation: 'Few-shotプロンプトは、期待する出力の例を少数示すことで、AIがパターンを理解しやすくする手法です。',
      },
      {
        id: 'q3',
        type: 'single',
        question: 'プロンプトの改善方法として不適切なものは？',
        options: [
          { id: 'a', text: '出力が不十分な場合、条件を追加する', isCorrect: false },
          { id: 'b', text: '対話を通じて徐々に精度を上げる', isCorrect: false },
          { id: 'c', text: '一度で完璧な出力を得るまでプロンプトを作り込む', isCorrect: true },
          { id: 'd', text: '具体的な例を示す', isCorrect: false },
        ],
        explanation: 'プロンプトは反復改善が基本です。一度で完璧を求めるよりも、対話的に改善していくことが効果的です。',
      },
    ],
  },
  {
    id: 'quiz-ai-practice',
    chapterId: 'ai-practice',
    title: 'AI活用演習 確認クイズ',
    description: '実践スキルを確認しましょう。',
    questions: [
      {
        id: 'q1',
        type: 'single',
        question: '「元の文章から重要な文をそのまま抜き出す」要約の種類は？',
        options: [
          { id: 'a', text: '圧縮型要約', isCorrect: false },
          { id: 'b', text: '抽出型要約', isCorrect: true },
          { id: 'c', text: '再構成型要約', isCorrect: false },
          { id: 'd', text: '翻訳型要約', isCorrect: false },
        ],
        explanation: '抽出型要約は、元の文章から重要な文をそのまま抜き出す手法です。',
      },
      {
        id: 'q2',
        type: 'single',
        question: 'AIによる校正で最も効果的なアプローチは？',
        options: [
          { id: 'a', text: '「校正して」とだけ指示する', isCorrect: false },
          { id: 'b', text: '校正のポイントと出力形式を具体的に指定する', isCorrect: true },
          { id: 'c', text: '校正は人間にしかできないのでAIは使わない', isCorrect: false },
          { id: 'd', text: '複数のAIに同時に依頼する', isCorrect: false },
        ],
        explanation: '校正のポイント（誤字脱字、文法、論理性など）と出力形式を具体的に指定することで、精度が向上します。',
      },
      {
        id: 'q3',
        type: 'single',
        question: '自己調整学習のサイクルとして正しい順序は？',
        options: [
          { id: 'a', text: '実行 → 計画 → 振り返り', isCorrect: false },
          { id: 'b', text: '計画 → 振り返り → 実行', isCorrect: false },
          { id: 'c', text: '計画 → 実行 → 振り返り', isCorrect: true },
          { id: 'd', text: '振り返り → 計画 → 実行', isCorrect: false },
        ],
        explanation: '自己調整学習は「計画→実行→振り返り」のサイクルで学習を進めます。',
      },
    ],
  },
  // ========== 第2段階：応用 ==========
  {
    id: 'quiz-cot',
    chapterId: 'cot',
    title: 'Chain-of-Thought手法 確認クイズ',
    description: 'CoTプロンプティングを確認しましょう。',
    questions: [
      {
        id: 'q1',
        type: 'single',
        question: 'CoTが回答精度を向上させる主な理由は？',
        options: [
          { id: 'a', text: 'AIの計算速度が上がるから', isCorrect: false },
          { id: 'b', text: '中間ステップを明示することで論理的な推論が促されるから', isCorrect: true },
          { id: 'c', text: 'インターネット検索が行われるから', isCorrect: false },
          { id: 'd', text: '複数のAIモデルが使用されるから', isCorrect: false },
        ],
        explanation: 'CoTは中間ステップを明示させることで、AIに論理的な推論プロセスを踏ませます。',
      },
      {
        id: 'q2',
        type: 'single',
        question: 'Zero-shot CoTの特徴として正しいものは？',
        options: [
          { id: 'a', text: '多数の例を示す必要がある', isCorrect: false },
          { id: 'b', text: '「ステップバイステップで考えてください」などの指示を追加するだけ', isCorrect: true },
          { id: 'c', text: '複数回の対話が必要', isCorrect: false },
          { id: 'd', text: '特定のドメインでしか使えない', isCorrect: false },
        ],
        explanation: 'Zero-shot CoTは例を示さずに「ステップバイステップで考えてください」という指示を追加するだけで実践できます。',
      },
      {
        id: 'q3',
        type: 'single',
        question: 'CoTが最も効果的なのは次のうちどれ？',
        options: [
          { id: 'a', text: '「今日の天気は？」という質問', isCorrect: false },
          { id: 'b', text: '「短い挨拶文を書いてください」という依頼', isCorrect: false },
          { id: 'c', text: '複数の条件を考慮した売上予測の計算', isCorrect: true },
          { id: 'd', text: '「赤いリンゴの画像を生成して」という依頼', isCorrect: false },
        ],
        explanation: 'CoTは複雑な問題、特に複数ステップの論理的推論が必要な問題で効果を発揮します。',
      },
    ],
  },
  {
    id: 'quiz-info-organization',
    chapterId: 'info-organization',
    title: '情報整理と論理的表現 確認クイズ',
    description: '情報整理の手法を確認しましょう。',
    questions: [
      {
        id: 'q1',
        type: 'single',
        question: 'MECEの原則として正しいものは？',
        options: [
          { id: 'a', text: '重要な項目だけを抽出する', isCorrect: false },
          { id: 'b', text: '漏れなく、重複なく分類する', isCorrect: true },
          { id: 'c', text: '時系列で整理する', isCorrect: false },
          { id: 'd', text: '優先度順に並べる', isCorrect: false },
        ],
        explanation: 'MECEは「Mutually Exclusive, Collectively Exhaustive」の略で、「漏れなく、重複なく」分類する原則です。',
      },
      {
        id: 'q2',
        type: 'single',
        question: 'ピラミッド構造で最上位に置くべきものは？',
        options: [
          { id: 'a', text: '詳細なデータ', isCorrect: false },
          { id: 'b', text: '背景説明', isCorrect: false },
          { id: 'c', text: '結論・メインメッセージ', isCorrect: true },
          { id: 'd', text: '問題提起', isCorrect: false },
        ],
        explanation: 'ピラミッド構造では、結論・メインメッセージを最上位に置き、その下に根拠を階層的に配置します。',
      },
      {
        id: 'q3',
        type: 'single',
        question: 'PREP法の正しい順序は？',
        options: [
          { id: 'a', text: '理由 → 結論 → 例 → まとめ', isCorrect: false },
          { id: 'b', text: '結論 → 例 → 理由 → まとめ', isCorrect: false },
          { id: 'c', text: '結論 → 理由 → 例 → 結論（再掲）', isCorrect: true },
          { id: 'd', text: '例 → 理由 → 結論 → まとめ', isCorrect: false },
        ],
        explanation: 'PREP法は Point（結論）→ Reason（理由）→ Example（例）→ Point（結論）の順です。',
      },
    ],
  },
  {
    id: 'quiz-project-management',
    chapterId: 'project-management',
    title: 'プロジェクト管理とAI 確認クイズ',
    description: 'プロジェクト管理の基礎を確認しましょう。',
    questions: [
      {
        id: 'q1',
        type: 'single',
        question: 'WBSの「100%ルール」とは？',
        options: [
          { id: 'a', text: 'すべてのタスクを100%完了させること', isCorrect: false },
          { id: 'b', text: '子要素の合計が親要素の100%を構成すること', isCorrect: true },
          { id: 'c', text: '品質を100%保証すること', isCorrect: false },
          { id: 'd', text: '100個以上のタスクに分解すること', isCorrect: false },
        ],
        explanation: 'WBSの100%ルールは、下位レベルの要素を合計すると上位レベルの100%を構成するという原則です。',
      },
      {
        id: 'q2',
        type: 'single',
        question: 'SMART原則の「M」が意味するものは？',
        options: [
          { id: 'a', text: 'Manageable（管理可能）', isCorrect: false },
          { id: 'b', text: 'Measurable（測定可能）', isCorrect: true },
          { id: 'c', text: 'Motivational（動機付け）', isCorrect: false },
          { id: 'd', text: 'Meaningful（意味のある）', isCorrect: false },
        ],
        explanation: 'SMART原則のMはMeasurable（測定可能）を意味し、目標の達成度を定量的に評価できることを指します。',
      },
      {
        id: 'q3',
        type: 'single',
        question: 'クリティカルパスの説明として正しいものは？',
        options: [
          { id: 'a', text: '最もコストがかかるタスクの経路', isCorrect: false },
          { id: 'b', text: '最もリスクが高いタスクの経路', isCorrect: false },
          { id: 'c', text: '遅延がプロジェクト全体の遅延に直結するタスクの経路', isCorrect: true },
          { id: 'd', text: '最も重要なタスクの経路', isCorrect: false },
        ],
        explanation: 'クリティカルパスは、遅延するとプロジェクト全体のスケジュールに影響する最長経路です。',
      },
    ],
  },
  {
    id: 'quiz-automation',
    chapterId: 'automation',
    title: '業務自動化ツール 確認クイズ',
    description: '自動化ツールの活用を確認しましょう。',
    questions: [
      {
        id: 'q1',
        type: 'single',
        question: '自動化に最も適した業務の特徴は？',
        options: [
          { id: 'a', text: '毎回異なる判断が必要な業務', isCorrect: false },
          { id: 'b', text: '定型的で繰り返し発生する業務', isCorrect: true },
          { id: 'c', text: '1回限りの特殊な業務', isCorrect: false },
          { id: 'd', text: '創造性が求められる業務', isCorrect: false },
        ],
        explanation: '自動化に適しているのは、定型的で繰り返し発生する業務です。',
      },
      {
        id: 'q2',
        type: 'single',
        question: 'Zapierで「自動化を起動するきっかけ」を何と呼ぶ？',
        options: [
          { id: 'a', text: 'Action', isCorrect: false },
          { id: 'b', text: 'Zap', isCorrect: false },
          { id: 'c', text: 'Trigger', isCorrect: true },
          { id: 'd', text: 'Filter', isCorrect: false },
        ],
        explanation: 'ZapierではTrigger（トリガー）が自動化を起動するきっかけとなるイベントです。',
      },
      {
        id: 'q3',
        type: 'single',
        question: 'Microsoft 365環境で最も適した自動化ツールは？',
        options: [
          { id: 'a', text: 'Zapier', isCorrect: false },
          { id: 'b', text: 'IFTTT', isCorrect: false },
          { id: 'c', text: 'Power Automate', isCorrect: true },
          { id: 'd', text: 'n8n', isCorrect: false },
        ],
        explanation: 'Power AutomateはMicrosoft製品との連携が優れており、Microsoft 365環境に最適です。',
      },
    ],
  },
  {
    id: 'quiz-business-problem',
    chapterId: 'business-problem',
    title: 'ビジネス課題解決 確認クイズ',
    description: '課題解決プロセスを確認しましょう。',
    questions: [
      {
        id: 'q1',
        type: 'single',
        question: 'ビジネス課題解決の正しい順序は？',
        options: [
          { id: 'a', text: '解決策→分析→定義→計画', isCorrect: false },
          { id: 'b', text: '定義→分析→原因特定→解決策→計画', isCorrect: true },
          { id: 'c', text: '分析→定義→解決策→原因特定→計画', isCorrect: false },
          { id: 'd', text: '計画→定義→分析→解決策', isCorrect: false },
        ],
        explanation: '課題解決は「定義→分析→原因特定→解決策→計画」の5ステップで進めます。',
      },
      {
        id: 'q2',
        type: 'single',
        question: '「なぜ」を繰り返して根本原因を探る手法を何と呼ぶ？',
        options: [
          { id: 'a', text: 'SWOT分析', isCorrect: false },
          { id: 'b', text: '5Why分析', isCorrect: true },
          { id: 'c', text: 'PEST分析', isCorrect: false },
          { id: 'd', text: '3C分析', isCorrect: false },
        ],
        explanation: '5Why分析は「なぜ？」を5回繰り返して根本原因を探る手法です。',
      },
      {
        id: 'q3',
        type: 'single',
        question: '経営層向け提案書で最初に置くべきものは？',
        options: [
          { id: 'a', text: '詳細なデータ分析', isCorrect: false },
          { id: 'b', text: '背景説明', isCorrect: false },
          { id: 'c', text: 'エグゼクティブサマリー', isCorrect: true },
          { id: 'd', text: '付録資料', isCorrect: false },
        ],
        explanation: '経営層は時間が限られているため、最初にエグゼクティブサマリー（要約）を置きます。',
      },
    ],
  },
  // ========== 第3段階：発展 ==========
  {
    id: 'quiz-paic',
    chapterId: 'paic',
    title: 'P-A-I-Cサイクル 確認クイズ',
    description: 'P-A-I-Cサイクルを確認しましょう。',
    questions: [
      {
        id: 'q1',
        type: 'single',
        question: 'P-A-I-Cサイクルの正しい順序は？',
        options: [
          { id: 'a', text: 'Ask → Prompt → Iterate → Create', isCorrect: false },
          { id: 'b', text: 'Prompt → Ask → Iterate → Create', isCorrect: true },
          { id: 'c', text: 'Create → Prompt → Ask → Iterate', isCorrect: false },
          { id: 'd', text: 'Prompt → Iterate → Ask → Create', isCorrect: false },
        ],
        explanation: 'P-A-I-Cは Prompt（準備）→ Ask（質問）→ Iterate（改善）→ Create（成果物）の順です。',
      },
      {
        id: 'q2',
        type: 'single',
        question: 'Iterateフェーズで最も重要なことは？',
        options: [
          { id: 'a', text: 'できるだけ多くの質問をする', isCorrect: false },
          { id: 'b', text: '具体的なフィードバックで改善を繰り返す', isCorrect: true },
          { id: 'c', text: 'AIの出力をそのまま受け入れる', isCorrect: false },
          { id: 'd', text: '新しいプロジェクトを開始する', isCorrect: false },
        ],
        explanation: 'Iterateフェーズでは、具体的なフィードバックを与えて出力を改善していきます。',
      },
      {
        id: 'q3',
        type: 'single',
        question: '成果物に含めるべきでないものは？',
        options: [
          { id: 'a', text: '本体（レポート、提案書等）', isCorrect: false },
          { id: 'b', text: 'プロセス記録', isCorrect: false },
          { id: 'c', text: '次のアクション', isCorrect: false },
          { id: 'd', text: 'AI出力の生データのみ', isCorrect: true },
        ],
        explanation: 'AI出力の生データだけでなく、人間による編集・検証・考察を加えた成果物にすべきです。',
      },
    ],
  },
  {
    id: 'quiz-pbl',
    chapterId: 'pbl',
    title: 'PBLプロジェクト設計 確認クイズ',
    description: 'PBLの基礎を確認しましょう。',
    questions: [
      {
        id: 'q1',
        type: 'single',
        question: 'PBL（課題解決型学習）の特徴として正しいものは？',
        options: [
          { id: 'a', text: '座学中心で知識を習得してから課題に取り組む', isCorrect: false },
          { id: 'b', text: '現実の課題に取り組みながら知識・スキルを習得する', isCorrect: true },
          { id: 'c', text: '個人作業が基本でチームワークは不要', isCorrect: false },
          { id: 'd', text: 'テストの点数で評価する', isCorrect: false },
        ],
        explanation: 'PBLは現実の課題に取り組みながら、必要な知識やスキルを身につけていく学習方法です。',
      },
      {
        id: 'q2',
        type: 'single',
        question: '良いプロジェクトテーマの条件として不適切なものは？',
        options: [
          { id: 'a', text: '現実の課題である', isCorrect: false },
          { id: 'b', text: 'AI活用の余地がある', isCorrect: false },
          { id: 'c', text: '1人で短時間に完結する', isCorrect: true },
          { id: 'd', text: 'チームで協働できる規模', isCorrect: false },
        ],
        explanation: 'PBLのテーマは、チームで協働でき、ある程度の規模と複雑さが必要です。',
      },
      {
        id: 'q3',
        type: 'single',
        question: '効果的なチームコラボレーションの方法は？',
        options: [
          { id: 'a', text: '週1回のミーティングだけで十分', isCorrect: false },
          { id: 'b', text: '日次の短い進捗共有と週次の振り返りを組み合わせる', isCorrect: true },
          { id: 'c', text: '困ったことがあっても自分で解決するまで報告しない', isCorrect: false },
          { id: 'd', text: '役割を厳密に分け、他の人の領域には関わらない', isCorrect: false },
        ],
        explanation: '日次の進捗共有と週次の振り返りを組み合わせることで、効果的なコラボレーションが実現できます。',
      },
    ],
  },
  {
    id: 'quiz-portfolio',
    chapterId: 'portfolio',
    title: 'ポートフォリオ作成 確認クイズ',
    description: 'ポートフォリオ作成を確認しましょう。',
    questions: [
      {
        id: 'q1',
        type: 'single',
        question: 'AI活用ポートフォリオの目的として不適切なものは？',
        options: [
          { id: 'a', text: '学習の証明', isCorrect: false },
          { id: 'b', text: '振り返りによる学びの深化', isCorrect: false },
          { id: 'c', text: 'スキルのアピール', isCorrect: false },
          { id: 'd', text: 'AIの出力をそのまま掲載する', isCorrect: true },
        ],
        explanation: 'ポートフォリオにはAI出力そのままではなく、自分の考察や工夫を加えた成果を掲載すべきです。',
      },
      {
        id: 'q2',
        type: 'single',
        question: 'プロジェクト紹介に含めるべきでないものは？',
        options: [
          { id: 'a', text: '背景と課題', isCorrect: false },
          { id: 'b', text: 'P-A-I-Cの記録', isCorrect: false },
          { id: 'c', text: '成果物と効果', isCorrect: false },
          { id: 'd', text: '失敗は隠して成功のみ記載', isCorrect: true },
        ],
        explanation: '失敗からの学びも重要な成果です。失敗とその対処を含めることで、より価値あるポートフォリオになります。',
      },
      {
        id: 'q3',
        type: 'single',
        question: 'ピアレビューの効果として正しいものは？',
        options: [
          { id: 'a', text: 'レビューを受ける側のみにメリットがある', isCorrect: false },
          { id: 'b', text: 'レビューする側の学びにもなる', isCorrect: true },
          { id: 'c', text: '競争意識を煽るためのもの', isCorrect: false },
          { id: 'd', text: 'ランキングをつけるためのもの', isCorrect: false },
        ],
        explanation: 'ピアレビューは、レビューする側も他者の視点から学ぶことができる相互学習の機会です。',
      },
    ],
  },
  {
    id: 'quiz-ethics-week',
    chapterId: 'ethics-week',
    title: 'AI倫理週間 確認クイズ',
    description: 'AI倫理の深い理解を確認しましょう。',
    questions: [
      {
        id: 'q1',
        type: 'single',
        question: '「人間中心のAI活用」の意味として正しいものは？',
        options: [
          { id: 'a', text: 'AIを使わないこと', isCorrect: false },
          { id: 'b', text: 'AIの判断を最終決定とすること', isCorrect: false },
          { id: 'c', text: '人間が主体となり、AIを道具として活用すること', isCorrect: true },
          { id: 'd', text: '人間の仕事をすべてAIに任せること', isCorrect: false },
        ],
        explanation: '人間中心のAI活用とは、人間が主体となり、AIを道具として活用することです。',
      },
      {
        id: 'q2',
        type: 'single',
        question: 'AIの公平性について正しい記述は？',
        options: [
          { id: 'a', text: 'AIは常に公平な判断をする', isCorrect: false },
          { id: 'b', text: 'データにバイアスがあるとAIも偏った判断をする', isCorrect: true },
          { id: 'c', text: '公平性の定義は一つしかない', isCorrect: false },
          { id: 'd', text: 'バイアスは技術的に完全に解消できる', isCorrect: false },
        ],
        explanation: 'AIは学習データに含まれるバイアスを反映するため、データにバイアスがあると偏った判断をします。',
      },
      {
        id: 'q3',
        type: 'single',
        question: '組織のAIガバナンスに含まれないものは？',
        options: [
          { id: 'a', text: 'AI活用の方針・原則', isCorrect: false },
          { id: 'b', text: '教育・啓発プログラム', isCorrect: false },
          { id: 'c', text: 'AIの技術的な性能向上', isCorrect: true },
          { id: 'd', text: 'インシデント対応手順', isCorrect: false },
        ],
        explanation: 'AIガバナンスは組織としてのAI活用の管理体制であり、技術的な性能向上は含まれません。',
      },
    ],
  },
  {
    id: 'quiz-self-coaching',
    chapterId: 'self-coaching',
    title: 'セルフコーチング 確認クイズ',
    description: 'セルフコーチング力を確認しましょう。',
    questions: [
      {
        id: 'q1',
        type: 'single',
        question: 'セルフコーチング力の構成要素として正しくないものは？',
        options: [
          { id: 'a', text: '自己認識', isCorrect: false },
          { id: 'b', text: '目標設定', isCorrect: false },
          { id: 'c', text: 'AIへの依存', isCorrect: true },
          { id: 'd', text: '振り返り', isCorrect: false },
        ],
        explanation: 'セルフコーチング力はAIへの依存ではなく、自律的に学び成長する力です。',
      },
      {
        id: 'q2',
        type: 'single',
        question: '3段階振り返りモデルの正しい順序は？',
        options: [
          { id: 'a', text: 'Move Forward → Look Back → Dig Deep', isCorrect: false },
          { id: 'b', text: 'Look Back → Dig Deep → Move Forward', isCorrect: true },
          { id: 'c', text: 'Dig Deep → Look Back → Move Forward', isCorrect: false },
          { id: 'd', text: 'Look Back → Move Forward → Dig Deep', isCorrect: false },
        ],
        explanation: '3段階振り返りモデルは Look Back（振り返る）→ Dig Deep（深掘りする）→ Move Forward（進む）の順です。',
      },
      {
        id: 'q3',
        type: 'single',
        question: 'SMART目標の「M」が意味するものは？',
        options: [
          { id: 'a', text: 'Manageable（管理可能）', isCorrect: false },
          { id: 'b', text: 'Measurable（測定可能）', isCorrect: true },
          { id: 'c', text: 'Meaningful（意味のある）', isCorrect: false },
          { id: 'd', text: 'Motivating（動機付けになる）', isCorrect: false },
        ],
        explanation: 'SMART目標のMはMeasurable（測定可能）を意味します。',
      },
    ],
  },
];

/**
 * 章IDから問題セットを取得
 */
export const getQuizByChapterId = (chapterId: string): Quiz | undefined => {
  return quizzes.find((quiz) => quiz.chapterId === chapterId);
};

/**
 * 問題セットIDから問題セットを取得
 */
export const getQuizById = (quizId: string): Quiz | undefined => {
  return quizzes.find((quiz) => quiz.id === quizId);
};

/**
 * 問題セットが存在する章IDの一覧を取得
 */
export const getChapterIdsWithQuiz = (): string[] => {
  return quizzes.map((quiz) => quiz.chapterId);
};
