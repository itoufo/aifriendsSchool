const DEFAULT_SITE_URL = 'https://ai-friends.school';

function normalizeSiteUrl(input: string): string {
  const trimmed = input.trim();
  if (!trimmed) return DEFAULT_SITE_URL;
  return trimmed.replace(/\/+$/, '');
}

export const SITE = {
  name: 'AI Friends School',
  title: 'AI Friends School - 包括的AIスクールカリキュラム',
  description:
    '初心者から経営者まで、5つのレベル別に体系的にAI活用能力を育成。真に「稼げる」人材、そして未来を創造できる人材になるための実践的教育プログラム。',
  // Use `NEXT_PUBLIC_SITE_URL` in production deployments.
  url: normalizeSiteUrl(process.env.NEXT_PUBLIC_SITE_URL ?? DEFAULT_SITE_URL),
  locale: 'ja_JP',
  language: 'ja',
  // Optional: set `NEXT_PUBLIC_TWITTER_HANDLE` like "@yourhandle"
  twitterHandle: process.env.NEXT_PUBLIC_TWITTER_HANDLE?.trim() || undefined,
} as const;

export function absoluteUrl(path: string): string {
  const base = SITE.url;
  if (!path) return base;
  if (path.startsWith('http://') || path.startsWith('https://')) return path;
  return `${base}${path.startsWith('/') ? path : `/${path}`}`;
}

