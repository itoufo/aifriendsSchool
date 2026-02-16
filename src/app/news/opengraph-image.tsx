import { ImageResponse } from 'next/og';
import { SITE } from '@/config/site';

export const runtime = 'edge';

export const alt = 'AIニュース';
export const size = {
  width: 1200,
  height: 630,
};
export const contentType = 'image/png';

export default function NewsOpenGraphImage() {
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
            'linear-gradient(135deg, #0b1220 0%, #111827 45%, #0284c7 130%)',
          padding: 64,
        }}
      >
        <div
          style={{
            width: '100%',
            maxWidth: 1020,
            display: 'flex',
            flexDirection: 'column',
            gap: 20,
            padding: 56,
            borderRadius: 32,
            background:
              'linear-gradient(135deg, rgba(255,255,255,0.12) 0%, rgba(255,255,255,0.06) 100%)',
            border: '1px solid rgba(255,255,255,0.16)',
          }}
        >
          <div
            style={{
              color: 'rgba(255,255,255,0.84)',
              fontSize: 26,
              fontWeight: 700,
              letterSpacing: 0.3,
            }}
          >
            {SITE.name}
          </div>

          <div
            style={{
              color: '#fff',
              fontSize: 70,
              fontWeight: 900,
              lineHeight: 1.08,
            }}
          >
            AIニュース・実務解説
          </div>

          <div
            style={{
              color: 'rgba(255,255,255,0.92)',
              fontSize: 30,
              lineHeight: 1.45,
            }}
          >
            最新トレンドを「実務でどう使うか」まで整理
          </div>

          <div
            style={{
              marginTop: 6,
              color: 'rgba(255,255,255,0.68)',
              fontSize: 22,
            }}
          >
            {SITE.url.replace(/^https?:\/\//, '')}/news
          </div>
        </div>
      </div>
    ),
    size
  );
}
