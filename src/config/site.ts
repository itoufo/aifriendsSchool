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
    '初心者から経営者まで学べる5レベル教材と、実務に効くAIニュース解説を提供。AI活用を体系的に学べる学習メディア。',
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
