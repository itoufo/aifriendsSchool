/**
 * 問題フォーマット仕様
 *
 * 選択問題形式の問題データを定義するための型定義
 * 各章に復習問題を紐付けることが可能
 */

/** 選択肢 */
export interface QuizOption {
  /** 選択肢ID (例: "a", "b", "c", "d") */
  id: string;
  /** 選択肢テキスト */
  text: string;
  /** 正解かどうか */
  isCorrect: boolean;
}

/** 問題 */
export interface QuizQuestion {
  /** 問題ID (例: "q1", "q2") */
  id: string;
  /** 問題タイプ: single=単一選択, multiple=複数選択 */
  type: "single" | "multiple";
  /** 問題文 */
  question: string;
  /** 選択肢の配列 */
  options: QuizOption[];
  /** 解説（回答後に表示） */
  explanation?: string;
}

/** 問題セット（章ごとの復習問題） */
export interface Quiz {
  /** 問題セットID */
  id: string;
  /** 対応する章のID (curriculum.tsのDocItem.idと対応) */
  chapterId: string;
  /** 問題セットのタイトル */
  title: string;
  /** 説明文 */
  description?: string;
  /** 問題の配列 */
  questions: QuizQuestion[];
}

/** ユーザーの回答状態 */
export interface QuizAnswer {
  /** 問題ID */
  questionId: string;
  /** 選択した選択肢ID（複数選択の場合は配列） */
  selectedOptionIds: string[];
  /** 正解かどうか */
  isCorrect?: boolean;
}

/** 問題セットの回答結果 */
export interface QuizResult {
  /** 問題セットID */
  quizId: string;
  /** 各問題の回答 */
  answers: QuizAnswer[];
  /** 正解数 */
  correctCount: number;
  /** 総問題数 */
  totalCount: number;
  /** スコア（パーセント） */
  score: number;
  /** 完了日時 */
  completedAt: Date;
}
