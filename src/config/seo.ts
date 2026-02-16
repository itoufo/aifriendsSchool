import type { DocItem, Section } from '@/data/curriculum';

export const BASE_SEO_KEYWORDS = [
  'AIスクール',
  'AI学習',
  'AI教育',
  '生成AI',
  'ChatGPT',
  'AIニュース',
  'AIメディア',
  'プロンプトエンジニアリング',
  'AI活用',
  'DX',
  'リスキリング',
  'オンライン学習',
  'AI Friends School',
];

export const SEO_FAQ_ITEMS = [
  {
    question: 'AI初心者でもAI Friends Schoolの学習についていけますか？',
    answer:
      'はい。スタートガイドとレベル1は、AIを初めて学ぶ方でも段階的に理解できる構成です。実務で使える内容を中心に、基礎から順番に学べます。',
  },
  {
    question: 'どのくらいの期間で実務に活かせるスキルが身につきますか？',
    answer:
      '目安はレベル1〜2で約2〜3か月です。業務効率化やプロンプト設計など、早い段階から現場で使えるスキルを習得できるように設計しています。',
  },
  {
    question: 'エンジニア以外でもAI活用を学べますか？',
    answer:
      '学べます。企画職・マーケター・管理職・経営層向けのカリキュラムを含み、職種ごとのAI活用方法を体系的に学習できます。',
  },
  {
    question: '学習後に収益化やキャリアに繋げる支援はありますか？',
    answer:
      'あります。収益化ロードマップ、ポートフォリオ構築、案件獲得ガイドを用意しており、副業や独立を見据えた実践的な学習が可能です。',
  },
] as const;

function dedupeKeywords(items: string[]): string[] {
  const normalized = new Set<string>();
  const output: string[] = [];

  for (const item of items) {
    const keyword = item.trim();
    if (!keyword) continue;
    const key = keyword.toLowerCase();
    if (normalized.has(key)) continue;
    normalized.add(key);
    output.push(keyword);
  }

  return output;
}

export function buildDocKeywords(doc: DocItem, section?: Section): string[] {
  return dedupeKeywords([
    ...BASE_SEO_KEYWORDS,
    doc.title,
    section?.title || '',
    section?.targetAudience || '',
    `${doc.title} 学習`,
    `${doc.title} 教材`,
    `${doc.title} 解説`,
  ]);
}

function stripMarkdown(raw: string): string {
  return raw
    .replace(/^---[\s\S]*?---\s*/m, '')
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/`[^`]*`/g, ' ')
    .replace(/!\[[^\]]*\]\([^)]+\)/g, ' ')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/<[^>]+>/g, ' ')
    .replace(/^[#>*-]\s+/gm, '')
    .replace(/^\d+\.\s+/gm, '')
    .replace(/\|/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

export function extractMarkdownSummary(markdown: string, maxLength = 150): string {
  const plain = stripMarkdown(markdown);
  if (!plain) return '';
  if (plain.length <= maxLength) return plain;
  return `${plain.slice(0, maxLength).trim()}...`;
}
