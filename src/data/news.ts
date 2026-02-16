export interface NewsSection {
  heading: string;
  paragraphs: string[];
}

export interface NewsArticle {
  slug: string;
  title: string;
  summary: string;
  category: string;
  categorySlug: string;
  tags: string[];
  publishedAt: string;
  updatedAt?: string;
  coverImage: string;
  sections: NewsSection[];
  relatedDocIds: string[];
}

export interface NewsCategory {
  slug: string;
  name: string;
  count: number;
}

export const newsArticles: NewsArticle[] = [
  {
    slug: 'ai-adoption-trends-2026-q1',
    title: '2026年Q1の生成AI導入トレンド: 現場で進む3つの実装パターン',
    summary:
      '企業のAI導入は「個人活用」から「業務フロー組み込み」へ。実装が進む3つの型と、導入時に外しやすい評価ポイントを解説します。',
    category: 'ニュース解説',
    categorySlug: 'news-analysis',
    tags: ['生成AI', '導入戦略', '業務効率化'],
    publishedAt: '2026-02-10',
    updatedAt: '2026-02-12',
    coverImage: '/images/illustrations/workflow-automation.jpg',
    relatedDocIds: ['business-efficiency', 'process-automation', 'ai-business-strategy'],
    sections: [
      {
        heading: '実装パターン1: 個人アシスタント型',
        paragraphs: [
          '最も導入しやすいのが、文章作成や要約、調査の下書きを支援する個人アシスタント型です。短期間で効果が出やすい反面、成果が個人に閉じると組織学習に繋がりにくい課題があります。',
          '導入初期はこの型から始め、活用ログや成功事例を蓄積し、次の業務フロー統合へ繋げる設計が重要です。',
        ],
      },
      {
        heading: '実装パターン2: チーム業務フロー型',
        paragraphs: [
          '次の段階では、議事録作成、提案資料ドラフト、問い合わせ一次対応など、チームで繰り返す作業にAIを組み込みます。成果は個人ではなくチームKPIで評価する必要があります。',
          'プロンプト標準化とレビュー基準の設計を先に固めることで、品質のばらつきを抑えられます。',
        ],
      },
      {
        heading: '実装パターン3: 部門横断オペレーション型',
        paragraphs: [
          '部門横断の型では、営業・マーケ・CS・開発が共通データを使い、AI活用を連携させます。この段階ではガバナンスとセキュリティ設計がROIを左右します。',
          '実装の鍵は、業務改善のKPIとAI品質KPIを分けて追うことです。速度だけでなく、誤答率・再修正率も同時に管理することで持続的に成果が出ます。',
        ],
      },
    ],
  },
  {
    slug: 'ai-agent-operations-checkpoints',
    title: 'AIエージェント運用で失敗しやすい5つのチェックポイント',
    summary:
      'PoCは成功したのに本番運用で失速するケースが増えています。AIエージェントを現場に定着させるために必要な5つの運用設計を整理します。',
    category: '運用ノウハウ',
    categorySlug: 'operations-knowhow',
    tags: ['AIエージェント', 'MLOps', '運用設計'],
    publishedAt: '2026-02-06',
    coverImage: '/images/illustrations/level4-production-systems.jpg',
    relatedDocIds: ['production-ai-systems', 'ai-security-privacy', 'ai-team-leadership'],
    sections: [
      {
        heading: '1. 役割定義が曖昧なまま導入してしまう',
        paragraphs: [
          'AIエージェントに任せる範囲と、人が最終判断すべき範囲を先に明確化しないと、責任境界が曖昧になります。',
          '業務単位で「自動化可能」「要承認」「人間判断必須」を分けるだけでも、運用時の混乱を大きく減らせます。',
        ],
      },
      {
        heading: '2. 成果指標が曖昧で改善サイクルが回らない',
        paragraphs: [
          '本番運用では精度だけでなく、処理時間、再作業率、顧客影響など複数指標で評価することが必要です。',
          'ダッシュボードを最初から作り込みすぎず、最小3指標で運用を始めると改善速度が上がります。',
        ],
      },
      {
        heading: '3. 例外処理フローを用意していない',
        paragraphs: [
          '例外時のエスカレーション設計がない場合、現場は結局すべて手作業に戻ります。',
          '失敗を前提に「どこで止めるか」「誰が引き取るか」を定義すると、運用の信頼性が上がります。',
        ],
      },
    ],
  },
  {
    slug: 'ai-news-research-framework-for-marketers',
    title: 'AIニュースをコンテンツマーケに変える情報収集フレーム',
    summary:
      'AIニュースを追うだけで終わらせず、見込み顧客獲得に繋げるための編集フローを紹介。ネタ選定から配信、再利用までを実務目線でまとめます。',
    category: 'コンテンツマーケ',
    categorySlug: 'content-marketing',
    tags: ['AIニュース', 'B2Bマーケティング', '編集設計'],
    publishedAt: '2026-01-31',
    coverImage: '/images/illustrations/client-acquisition.jpg',
    relatedDocIds: ['business-planning-communication', 'client-acquisition', 'monetization-roadmap'],
    sections: [
      {
        heading: 'ニュースを「自社文脈」に翻訳する',
        paragraphs: [
          'ニュース単体では差別化が難しいため、顧客課題との接点を定義してから記事化することが重要です。',
          '例えば「モデル更新」そのものではなく、「自社の業務で何が変わるか」を見出しにすると、読者の行動に繋がります。',
        ],
      },
      {
        heading: '1テーマを複数フォーマットに再利用する',
        paragraphs: [
          '1本のニュース解説を、長文記事、短尺SNS投稿、営業資料の補足に再利用すると、制作効率が大きく上がります。',
          '企画段階で再利用先を決めておくと、編集体制が安定しやすくなります。',
        ],
      },
      {
        heading: '問い合わせ導線まで設計する',
        paragraphs: [
          'ニュース記事は認知だけでなく、相談導線に繋がるCTAを設置して初めて商談化しやすくなります。',
          '診断コンテンツ、チェックリスト、無料相談など、読者の温度感に合わせたCTAを複数用意するのが効果的です。',
        ],
      },
    ],
  },
  {
    slug: 'rag-kpi-design-practical-guide',
    title: 'RAGプロジェクトのKPI設計: 精度だけでは失敗する理由',
    summary:
      'RAG導入の成否は、検索精度だけでなく業務成果指標とセットで設計できるかで決まります。現場で使えるKPI設計テンプレートを紹介します。',
    category: '技術実務',
    categorySlug: 'technical-practice',
    tags: ['RAG', 'KPI', 'ナレッジ活用'],
    publishedAt: '2026-01-24',
    coverImage: '/images/illustrations/data-analysis.jpg',
    relatedDocIds: ['ai-infrastructure-architecture', 'data-analysis-insights', 'production-ai-systems'],
    sections: [
      {
        heading: 'KPIは3階層で分ける',
        paragraphs: [
          'RAGでは「検索品質」「回答品質」「業務成果」を分けて追う必要があります。検索精度が高くても業務成果が出ないケースは珍しくありません。',
          '導入初期は、回答到達時間と再問い合わせ率を最小セットとして計測すると改善の方向性が見えます。',
        ],
      },
      {
        heading: '評価データを業務実態に合わせる',
        paragraphs: [
          '評価用データが理想的すぎると、本番で期待値を下回りやすくなります。実際の問い合わせや文書を反映したテストセットを用意することが重要です。',
          '月次で評価データを更新し、業務変化に追随させることで、劣化の早期検知が可能になります。',
        ],
      },
    ],
  },
  {
    slug: 'ai-governance-quick-checklist-2026',
    title: '2026年版 AIガバナンス実務チェックリスト',
    summary:
      'AI活用が全社展開されるほど、ガバナンス設計は後回しにできません。現場で使える実務チェックリストを6項目で整理しました。',
    category: 'ガバナンス',
    categorySlug: 'governance',
    tags: ['AIガバナンス', 'リスク管理', 'コンプライアンス'],
    publishedAt: '2026-01-18',
    coverImage: '/images/illustrations/level5-governance.jpg',
    relatedDocIds: ['ai-governance-ethics', 'ai-security-privacy', 'organization-culture-change'],
    sections: [
      {
        heading: '運用責任者と意思決定フローを定義する',
        paragraphs: [
          '責任者不在のままツール導入だけが進むと、事故発生時に対応が遅れます。最低限、意思決定者と承認ルートを文書化しておく必要があります。',
        ],
      },
      {
        heading: 'データ利用ルールを明文化する',
        paragraphs: [
          '社内データを扱う場合は、入力禁止情報、マスキング基準、ログ保存方針を明確化し、利用者教育とセットで運用します。',
        ],
      },
      {
        heading: '監査可能な運用ログを残す',
        paragraphs: [
          'モデル更新やプロンプト変更、ワークフロー変更を追跡可能にしておくと、問題発生時の再発防止がしやすくなります。',
        ],
      },
    ],
  },
];

function getSortDate(article: NewsArticle): number {
  return new Date(article.updatedAt || article.publishedAt).getTime();
}

function getTagOverlapScore(source: NewsArticle, candidate: NewsArticle): number {
  const sourceTags = new Set(source.tags.map((tag) => tag.toLowerCase()));
  return candidate.tags.reduce((score, tag) => {
    return sourceTags.has(tag.toLowerCase()) ? score + 1 : score;
  }, 0);
}

function getTextLength(article: NewsArticle): number {
  const sectionText = article.sections
    .flatMap((section) => [section.heading, ...section.paragraphs])
    .join('');
  return (article.title + article.summary + sectionText).length;
}

export function getAllNewsArticles(): NewsArticle[] {
  return [...newsArticles].sort((a, b) => getSortDate(b) - getSortDate(a));
}

export function getNewsBySlug(slug: string): NewsArticle | undefined {
  return newsArticles.find((article) => article.slug === slug);
}

export function getLatestNews(limit = 3): NewsArticle[] {
  return getAllNewsArticles().slice(0, limit);
}

export function getNewsByCategorySlug(categorySlug: string): NewsArticle[] {
  return getAllNewsArticles().filter((article) => article.categorySlug === categorySlug);
}

export function getAllNewsCategories(): NewsCategory[] {
  const counts = new Map<string, NewsCategory>();

  for (const article of newsArticles) {
    const existing = counts.get(article.categorySlug);
    if (existing) {
      existing.count += 1;
      continue;
    }
    counts.set(article.categorySlug, {
      slug: article.categorySlug,
      name: article.category,
      count: 1,
    });
  }

  return [...counts.values()].sort((a, b) => {
    if (b.count !== a.count) return b.count - a.count;
    return a.name.localeCompare(b.name, 'ja');
  });
}

export function getCategoryNameBySlug(categorySlug: string): string | undefined {
  return newsArticles.find((article) => article.categorySlug === categorySlug)?.category;
}

export function getRelatedNewsArticles(sourceSlug: string, limit = 3): NewsArticle[] {
  const source = getNewsBySlug(sourceSlug);
  if (!source) return [];

  const ranked = getAllNewsArticles()
    .filter((candidate) => candidate.slug !== source.slug)
    .map((candidate) => {
      const tagScore = getTagOverlapScore(source, candidate) * 2;
      const categoryScore = candidate.categorySlug === source.categorySlug ? 3 : 0;
      const recencyBoost = Math.max(
        0,
        1 -
          Math.min(
            1,
            (Date.now() - getSortDate(candidate)) / (1000 * 60 * 60 * 24 * 90)
          )
      );
      return {
        article: candidate,
        score: tagScore + categoryScore + recencyBoost,
      };
    })
    .sort((a, b) => {
      if (b.score !== a.score) return b.score - a.score;
      return getSortDate(b.article) - getSortDate(a.article);
    });

  return ranked.slice(0, limit).map((entry) => entry.article);
}

export function estimateNewsReadingMinutes(article: NewsArticle): number {
  const textLength = getTextLength(article);
  const charsPerMinute = 650;
  return Math.max(2, Math.ceil(textLength / charsPerMinute));
}

export function formatNewsDate(dateValue: string): string {
  return new Intl.DateTimeFormat('ja-JP', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  }).format(new Date(dateValue));
}
