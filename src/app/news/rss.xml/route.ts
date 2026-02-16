import { absoluteUrl, SITE } from '@/config/site';
import { getAllNewsArticles } from '@/data/news';

function escapeXml(value: string): string {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');
}

export function GET() {
  const articles = getAllNewsArticles();
  const lastBuildDate =
    articles[0]?.updatedAt || articles[0]?.publishedAt || new Date().toISOString();

  const items = articles
    .map((article) => {
      const url = absoluteUrl(`/news/${article.slug}`);
      const pubDate = new Date(article.updatedAt || article.publishedAt).toUTCString();

      return `
        <item>
          <title>${escapeXml(article.title)}</title>
          <link>${escapeXml(url)}</link>
          <guid isPermaLink="true">${escapeXml(url)}</guid>
          <pubDate>${pubDate}</pubDate>
          <description>${escapeXml(article.summary)}</description>
          <category>${escapeXml(article.category)}</category>
        </item>
      `;
    })
    .join('');

  const xml = `<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>${escapeXml(`${SITE.name} AIニュース`)}</title>
    <link>${escapeXml(absoluteUrl('/news'))}</link>
    <description>${escapeXml(
      '生成AIの最新トレンド、導入実務、運用ノウハウを分かりやすく解説するニュースフィード'
    )}</description>
    <language>ja</language>
    <lastBuildDate>${new Date(lastBuildDate).toUTCString()}</lastBuildDate>
    ${items}
  </channel>
</rss>`;

  return new Response(xml, {
    headers: {
      'Content-Type': 'application/rss+xml; charset=utf-8',
      'Cache-Control': 'public, s-maxage=3600, stale-while-revalidate=86400',
    },
  });
}
