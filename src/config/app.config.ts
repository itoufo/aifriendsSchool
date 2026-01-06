/**
 * アプリケーション設定
 */
export const AppConfig = {
  // アプリケーション基本情報
  app: {
    name: 'AI Friends School',
    version: '1.0.0',
    description: '包括的AIスクールカリキュラム - 初心者から経営者まで5段階の体系的AI教育',
  },

  // パス設定
  paths: {
    docs: '/docs',           // Markdownドキュメントのパス
    quizzes: '/quizzes',     // クイズJSONファイルのパス
    images: '/docs/images',  // 画像ファイルのパス
  },

  // クイズ設定
  quiz: {
    passThreshold: 70,       // 合格ライン（%）
    enableCache: true,       // クイズのキャッシュを有効化
    showExplanation: true,   // 解説を表示
  },

  // 進捗管理設定
  progress: {
    storageKey: 'aifriends-school-progress',
    statsKey: 'aifriends-school-stats',
    autoSave: true,
  },

  // ノート設定
  notes: {
    storageKey: 'aifriends-school-notes',
    bookmarksKey: 'aifriends-school-bookmarks',
    maxNoteLength: 5000,     // ノートの最大文字数
  },

  // ブランディング設定
  branding: {
    title: 'AI Friends School - 包括的AIスクールカリキュラム',       // トップバーのタイトル
    logo: {
      src: '/images/logo.png',  // ロゴ画像のパス
      alt: 'AIフレンズ',         // ロゴのalt属性
    },
    footer: {
      text: '© 2024 AI Friends School - 真に稼げる人材、そして未来を創造できる人材を育成',
    },
  },

  // UI設定
  ui: {
    sidebarWidth: 280,       // サイドバーの幅（px）
    enableAnimations: true,  // アニメーションを有効化
    theme: 'light',         // テーマ（light/dark）
  },
};