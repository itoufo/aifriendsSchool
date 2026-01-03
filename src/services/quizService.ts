import type { Quiz } from '../data/quiz.types';
import { AppConfig } from '../config/app.config';

/**
 * クイズJSONファイルを動的に読み込むサービス
 */
export class QuizService {
  private static cache = new Map<string, Quiz>();

  /**
   * 章IDに対応するクイズを読み込む
   * @param docId 章のID
   * @returns クイズデータまたはnull
   */
  static async loadQuiz(docId: string): Promise<Quiz | null> {
    // キャッシュチェック
    if (this.cache.has(docId)) {
      return this.cache.get(docId) || null;
    }

    try {
      const response = await fetch(`${AppConfig.paths.quizzes}/${docId}.json`);
      
      if (!response.ok) {
        // クイズファイルが存在しない場合はnullを返す
        if (response.status === 404) {
          console.log(`Quiz not found for chapter: ${docId}`);
          return null;
        }
        throw new Error(`Failed to load quiz: ${response.status}`);
      }

      const quiz: Quiz = await response.json();
      
      // キャッシュに保存
      this.cache.set(docId, quiz);
      
      return quiz;
    } catch (error) {
      console.error(`Error loading quiz for ${docId}:`, error);
      return null;
    }
  }

  /**
   * 複数のクイズを一括で読み込む
   * @param docIds 章のID配列
   * @returns クイズデータの配列
   */
  static async loadMultipleQuizzes(docIds: string[]): Promise<(Quiz | null)[]> {
    const promises = docIds.map(id => this.loadQuiz(id));
    return Promise.all(promises);
  }

  /**
   * クイズの存在確認
   * @param docId 章のID
   * @returns クイズが存在するかどうか
   */
  static async hasQuiz(docId: string): Promise<boolean> {
    const quiz = await this.loadQuiz(docId);
    return quiz !== null;
  }

  /**
   * キャッシュをクリア
   */
  static clearCache(): void {
    this.cache.clear();
  }

  /**
   * 利用可能なすべてのクイズIDを取得
   * @returns クイズIDの配列
   */
  static async getAvailableQuizzes(): Promise<string[]> {
    try {
      // クイズディレクトリのインデックスファイルを読み込む
      const response = await fetch(`${AppConfig.paths.quizzes}/index.json`);
      if (!response.ok) {
        return [];
      }
      const index = await response.json();
      return index.quizzes || [];
    } catch (error) {
      console.error('Error loading quiz index:', error);
      return [];
    }
  }
}