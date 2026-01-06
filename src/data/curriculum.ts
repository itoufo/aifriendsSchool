export interface DocItem {
  id: string;
  title: string;
  path: string;
  duration?: string; // 学習時間の目安
  level?: string; // 対象レベル
}

export interface Section {
  id: string;
  title: string;
  description?: string; // セクションの説明
  targetAudience?: string; // 対象者
  duration?: string; // 期間目安
  items: DocItem[];
}

export const curriculum: Section[] = [
  {
    id: 'level-0-start',
    title: 'スタートガイド',
    description: 'AI Schoolへようこそ！AIで稼ぐための第一歩を踏み出しましょう。学習の準備からAIサービスの使い方まで、ここから始めます。',
    targetAudience: 'これからAIを学び始める全ての方、副業・独立を目指す方',
    duration: '1-2日（準備期間）',
    items: [
      {
        id: 'getting-started',
        title: 'はじめに：稼げるAI人材への道',
        path: '/docs/level0/00_getting-started.md',
        duration: '30分',
        level: 'start'
      },
      {
        id: 'ai-services-guide',
        title: 'AIサービス完全ガイド',
        path: '/docs/level0/01_ai-services-guide.md',
        duration: '1時間',
        level: 'start'
      }
    ]
  },
  {
    id: 'level-1-beginner',
    title: 'レベル1：初心者',
    description: 'AIの基本概念を完全に理解し、日常業務や学習において、複数のAIツールを自律的に、かつ、倫理的に正しく活用できる状態になる。',
    targetAudience: 'AIに初めて触れる、または知識が断片的なビジネスパーソン、学生',
    duration: '4週間（週10時間程度の学習を想定）',
    items: [
      { 
        id: 'ai-literacy-and-ethics', 
        title: 'AIリテラシーと倫理', 
        path: '/docs/level1/01_ai-literacy-and-ethics.md',
        duration: '1週目',
        level: 'beginner'
      },
      { 
        id: 'prompt-engineering-basics', 
        title: 'プロンプトエンジニアリング入門', 
        path: '/docs/level1/02_prompt-engineering-basics.md',
        duration: '2週目',
        level: 'beginner'
      },
      { 
        id: 'business-efficiency', 
        title: '業務効率化の実践', 
        path: '/docs/level1/03_business-efficiency.md',
        duration: '3週目',
        level: 'beginner'
      },
      {
        id: 'human-skills-integration',
        title: 'ヒューマンスキルとの融合',
        path: '/docs/level1/04_human-skills-integration.md',
        duration: '4週目',
        level: 'beginner'
      },
      {
        id: 'hands-on-exercises-1',
        title: '実践演習：手を動かして学ぶ',
        path: '/docs/level1/05_hands-on-exercises.md',
        duration: '5週目',
        level: 'beginner'
      }
    ],
  },
  {
    id: 'level-2-intermediate',
    title: 'レベル2：中級者',
    description: '担当業務におけるボトルネックを自ら発見し、複数のAIツールとiPaaSなどを連携させて課題解決やプロセス改善を自律的に実行できる。また、その成果を定量的に示し、チームに共有できる。',
    targetAudience: 'AIツールの基本操作はできるが、より体系的な活用法を学びたい企画職、マーケターなど',
    duration: '6週間（週10時間程度の学習を想定）',
    items: [
      { 
        id: 'advanced-prompt-engineering', 
        title: '高度なプロンプトエンジニアリング', 
        path: '/docs/level2/01_advanced-prompt-engineering.md',
        duration: '1週目',
        level: 'intermediate'
      },
      { 
        id: 'data-analysis-insights', 
        title: 'データ分析とインサイト抽出', 
        path: '/docs/level2/02_data-analysis-insights.md',
        duration: '2-3週目',
        level: 'intermediate'
      },
      { 
        id: 'process-automation', 
        title: '業務プロセスの自動化', 
        path: '/docs/level2/03_process-automation.md',
        duration: '4-5週目',
        level: 'intermediate'
      },
      {
        id: 'business-planning-communication',
        title: 'AI時代のビジネス企画と発信',
        path: '/docs/level2/04_business-planning-communication.md',
        duration: '6週目',
        level: 'intermediate'
      },
      {
        id: 'hands-on-exercises-2',
        title: '実践演習：案件レベルの実践',
        path: '/docs/level2/05_hands-on-exercises.md',
        duration: '7週目',
        level: 'intermediate'
      }
    ],
  },
  {
    id: 'level-3-advanced',
    title: 'レベル3：上級者',
    description: '部署やチームのビジネス課題に対し、AIを活用した解決策をプロジェクトとして企画・推進できる。ROIを意識した上で、適切なAI技術・ツールを選定し、チームの生産性を最大化するためのAIガバナンスと組織変革を主導できる。',
    targetAudience: '部署やチームのDXを推進するリーダー、マネージャー',
    duration: '8週間（週10時間程度の学習を想定）',
    items: [
      { 
        id: 'ai-project-management', 
        title: 'AIプロジェクトマネジメント', 
        path: '/docs/level3/01_ai-project-management.md',
        duration: '1-2週目',
        level: 'advanced'
      },
      { 
        id: 'custom-ai-vibes-coding', 
        title: 'カスタムAIとVibesコーディングの理解', 
        path: '/docs/level3/02_custom-ai-vibes-coding.md',
        duration: '3-4週目',
        level: 'advanced'
      },
      { 
        id: 'ai-governance-organization', 
        title: 'AIガバナンスと組織導入', 
        path: '/docs/level3/03_ai-governance-organization.md',
        duration: '5-6週目',
        level: 'advanced'
      },
      { 
        id: 'business-strategy-leadership', 
        title: 'ビジネス戦略とリーダーシップ', 
        path: '/docs/level3/04_business-strategy-leadership.md',
        duration: '7-8週目',
        level: 'advanced'
      },
    ],
  },
  {
    id: 'level-4-engineer',
    title: 'エンジニア',
    description: 'AIを第一の選択肢として捉え、従来の開発手法とAIネイティブな開発手法（Vibesコーディング、AIエージェント開発など）をハイブリッドに使いこなせる。ビジネス要件を深く理解し、最適なAIモデルの選定・ファインチューニングから、スケーラブルで信頼性の高いAIアプリケーションの実装、MLOpsパイプラインの構築までをエンドツーエンドで担当できる技術的リーダーになる。',
    targetAudience: 'AI技術を自身の専門分野に活かしたいITエンジニア、開発者',
    duration: '10週間（週10-15時間程度の学習を想定）',
    items: [
      { 
        id: 'ai-native-development-environment', 
        title: 'AIネイティブ開発環境と新パラダイム', 
        path: '/docs/level4/01_ai-native-development-environment.md',
        duration: '1-2週目',
        level: 'engineer'
      },
      { 
        id: 'machine-learning-mlops', 
        title: '実践・機械学習とMLOps', 
        path: '/docs/level4/02_machine-learning-mlops.md',
        duration: '3-4週目',
        level: 'engineer'
      },
      { 
        id: 'generative-ai-applications', 
        title: '生成AIアプリケーション開発', 
        path: '/docs/level4/03_generative-ai-applications.md',
        duration: '5-7週目',
        level: 'engineer'
      },
      { 
        id: 'business-technology-bridge', 
        title: 'ビジネスと技術の架け橋', 
        path: '/docs/level4/04_business-technology-bridge.md',
        duration: '8-10週目',
        level: 'engineer'
      },
    ],
  },
  {
    id: 'level-5-executive',
    title: '経営者',
    description: 'AIの本質とビジネスインパクトを深く理解し、自社の経営戦略とAI戦略を完全に統合させる。AIを単なる効率化ツールではなく、持続的な競争優位性を築き、新たな事業価値を創造するための根源として捉え、全社的な変革を主導するビジョンとリーダーシップを発揮できる。',
    targetAudience: '企業のDXやAI戦略を担う経営層、事業責任者',
    duration: '4週間（週8時間の学習・ディスカッションを想定。ケーススタディ中心）',
    items: [
      { 
        id: 'ai-business-strategy-integration', 
        title: 'AIと経営戦略の統合', 
        path: '/docs/level5/01_ai-business-strategy-integration.md',
        duration: '1週目',
        level: 'executive'
      },
      { 
        id: 'ai-era-organization-talent', 
        title: 'AI時代の組織と人材', 
        path: '/docs/level5/02_ai-era-organization-talent.md',
        duration: '2週目',
        level: 'executive'
      },
      { 
        id: 'ai-investment-risk-management', 
        title: 'AI投資とリスクマネジメント', 
        path: '/docs/level5/03_ai-investment-risk-management.md',
        duration: '3週目',
        level: 'executive'
      },
      {
        id: 'vision-leadership',
        title: 'ビジョンとリーダーシップ',
        path: '/docs/level5/04_vision-leadership.md',
        duration: '4週目',
        level: 'executive'
      },
    ],
  },
  {
    id: 'career-monetization',
    title: '収益化・キャリア',
    description: '学んだスキルを収益に変える具体的な方法。副業から独立まで、あなたの目標に合わせたロードマップを提供します。',
    targetAudience: '副業・フリーランス・独立を目指す全ての方',
    duration: '随時参照',
    items: [
      {
        id: 'monetization-roadmap',
        title: '収益化ロードマップ',
        path: '/docs/career/01_monetization-roadmap.md',
        duration: '45分',
        level: 'career'
      },
      {
        id: 'portfolio-guide',
        title: 'ポートフォリオ構築ガイド',
        path: '/docs/career/02_portfolio-guide.md',
        duration: '60分',
        level: 'career'
      },
      {
        id: 'client-acquisition',
        title: '案件獲得実践ガイド',
        path: '/docs/career/03_client-acquisition.md',
        duration: '90分',
        level: 'career'
      }
    ]
  },
];

export const getDocByPath = (path: string): DocItem | undefined => {
  for (const section of curriculum) {
    const item = section.items.find((item) => item.path === path);
    if (item) return item;
  }
  return undefined;
};

export const getDocById = (id: string): DocItem | undefined => {
  for (const section of curriculum) {
    const item = section.items.find((item) => item.id === id);
    if (item) return item;
  }
  return undefined;
};

/**
 * 指定された章IDの次の章を取得
 */
export const getNextDoc = (currentId: string): DocItem | undefined => {
  const allDocs: DocItem[] = [];
  for (const section of curriculum) {
    allDocs.push(...section.items);
  }

  const currentIndex = allDocs.findIndex((doc) => doc.id === currentId);
  if (currentIndex === -1 || currentIndex === allDocs.length - 1) {
    return undefined;
  }

  return allDocs[currentIndex + 1];
};
