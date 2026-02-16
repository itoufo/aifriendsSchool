import { ImageResponse } from 'next/og';
import { SITE } from '@/config/site';
import { getNewsBySlug } from '@/data/news';

export const runtime = 'edge';

export const alt = 'AIニュース記事';
export const size = {
  width: 1200,
  height: 630,
};
export const contentType = 'image/png';

export default function NewsArticleOpenGraphImage({
  params,
}: {
  params: { slug: string };
}) {
  const article = getNewsBySlug(params.slug);
  const title = article?.title ?? 'AIニュース';
  const category = article?.category ?? 'ニュース解説';

  return new ImageResponse(
    (
      <div
        style={{
          width: '100%',
          height: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          background:
            'linear-gradient(135deg, #0b1220 0%, #111827 45%, #0ea5e9 145%)',
          padding: 64,
        }}
      >
        <div
          style={{
            width: '100%',
            maxWidth: 1020,
            display: 'flex',
            flexDirection: 'column',
            gap: 18,
            padding: 54,
            borderRadius: 30,
            background:
              'linear-gradient(135deg, rgba(255,255,255,0.12) 0%, rgba(255,255,255,0.06) 100%)',
            border: '1px solid rgba(255,255,255,0.16)',
          }}
        >
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              gap: 12,
            }}
          >
            <div
              style={{
                color: 'rgba(255,255,255,0.82)',
                fontSize: 24,
                fontWeight: 700,
              }}
            >
              {SITE.name}
            </div>
            <div
              style={{
                color: 'rgba(255,255,255,0.82)',
                fontSize: 20,
                fontWeight: 700,
              }}
            >
              {category}
            </div>
          </div>

          <div
            style={{
              color: '#fff',
              fontSize: 56,
              fontWeight: 900,
              lineHeight: 1.15,
              wordBreak: 'break-word',
            }}
          >
            {title}
          </div>

          <div
            style={{
              marginTop: 6,
              color: 'rgba(255,255,255,0.72)',
              fontSize: 22,
            }}
          >
            {SITE.url.replace(/^https?:\/\//, '')}/news/{params.slug}
          </div>
        </div>
      </div>
    ),
    size
  );
}
